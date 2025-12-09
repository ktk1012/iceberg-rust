// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Apply name mapping to Arrow schema to assign Iceberg field IDs.
//!
//! This module implements the `ApplyNameMapping` visitor that traverses an Arrow schema
//! and assigns field IDs from an Iceberg `NameMapping`. This is necessary when reading
//! Parquet files that lack field IDs (e.g., files migrated from Hive/Spark).
//!
//! The implementation uses `ArrowSchemaVisitor` from schema.rs with a stack-based
//! context tracking, following the same design as Java's `ApplyNameMapping` visitor.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use super::schema::{ArrowSchemaVisitor, visit_schema};
use crate::error::Result;
use crate::spec::{MappedField, NameMapping};

/// Normalized field name for list elements (matches Java implementation).
const LIST_ELEMENT_NAME: &str = "element";
/// Normalized field name for map keys (matches Java implementation).
const MAP_KEY_NAME: &str = "key";
/// Normalized field name for map values (matches Java implementation).
const MAP_VALUE_NAME: &str = "value";

/// Stores field information for stack-based traversal.
/// This is needed because `ArrowSchemaVisitor::primitive()` only receives
/// DataType, not the full Field information.
struct FieldInfo {
    name: String,
    nullable: bool,
    metadata: HashMap<String, String>,
    /// The lookup name used for name mapping (may differ from actual name
    /// for list elements, map keys/values which use normalized names).
    lookup_name: String,
}

impl FieldInfo {
    fn new(field: &Field, lookup_name: &str) -> Self {
        Self {
            name: field.name().clone(),
            nullable: field.is_nullable(),
            metadata: field.metadata().clone(),
            lookup_name: lookup_name.to_string(),
        }
    }
}

/// Visitor that applies a `NameMapping` to an Arrow schema using `ArrowSchemaVisitor`.
///
/// This visitor traverses the Arrow schema and assigns field IDs
/// from the name mapping based on field names. It handles nested types
/// (struct, list, map) by maintaining a stack of mapped fields for
/// proper context tracking.
///
/// The implementation follows Java's `ApplyNameMapping` pattern:
/// - Uses a stack to track the current mapping context during traversal
/// - Uses normalized names ("element", "key", "value") for list/map children
/// - Supports field name aliases through `MappedField.names()`
struct ApplyNameMappingVisitor<'a> {
    /// Stack of current mapped fields for nested traversal.
    /// Each entry represents the children mappings at that level.
    mapping_stack: Vec<&'a [Arc<MappedField>]>,
    /// Stack of field information for reconstructing Fields in visitor methods.
    field_info_stack: Vec<FieldInfo>,
}

impl<'a> ApplyNameMappingVisitor<'a> {
    /// Create a new visitor with the adapter's root fields.
    fn new(root_fields: &'a [Arc<MappedField>]) -> Self {
        Self {
            mapping_stack: vec![root_fields],
            field_info_stack: Vec::new(),
        }
    }

    /// Get the current mapping context (top of the stack).
    fn current_mapping(&self) -> &'a [Arc<MappedField>] {
        self.mapping_stack.last().copied().unwrap_or(&[])
    }

    /// Find a mapped field by name in the current context.
    fn find_in_current(&self, name: &str) -> Option<&'a Arc<MappedField>> {
        self.current_mapping()
            .iter()
            .find(|mf| mf.names().iter().any(|n| n == name))
    }

    /// Push children of a mapped field onto the stack.
    fn push_mapping_context(&mut self, mapped_field: Option<&'a Arc<MappedField>>) {
        let children = mapped_field.map(|mf| mf.fields()).unwrap_or(&[]);
        self.mapping_stack.push(children);
    }

    /// Pop the current mapping context from the stack.
    fn pop_mapping_context(&mut self) {
        self.mapping_stack.pop();
    }

    /// Push field info onto the stack and update mapping context.
    fn push_field_context(&mut self, field: &Field, lookup_name: &str) {
        self.field_info_stack
            .push(FieldInfo::new(field, lookup_name));
        let mapped = self.find_in_current(lookup_name);
        self.push_mapping_context(mapped);
    }

    /// Pop field info and mapping context from stacks.
    fn pop_field_context(&mut self) {
        self.field_info_stack.pop();
        self.pop_mapping_context();
    }

    /// Get current field info (top of stack).
    fn current_field_info(&self) -> Option<&FieldInfo> {
        self.field_info_stack.last()
    }

    /// Apply field ID to a field based on the mapping found for the given lookup name.
    fn apply_field_id(&self, field: Field, lookup_name: &str) -> Field {
        // Look up in the parent's mapping context (one level up from current)
        let parent_mapping = if self.mapping_stack.len() >= 2 {
            self.mapping_stack[self.mapping_stack.len() - 2]
        } else {
            self.mapping_stack.last().copied().unwrap_or(&[])
        };

        let mapped = parent_mapping
            .iter()
            .find(|mf| mf.names().iter().any(|n| n == lookup_name));

        if let Some(mf) = mapped {
            if let Some(field_id) = mf.field_id() {
                let mut metadata = field.metadata().clone();
                metadata.insert(PARQUET_FIELD_ID_META_KEY.to_string(), field_id.to_string());
                return Field::new(field.name(), field.data_type().clone(), field.is_nullable())
                    .with_metadata(metadata);
            }
        }

        field
    }
}

impl<'a> ArrowSchemaVisitor for ApplyNameMappingVisitor<'a> {
    type T = Field;
    type U = ArrowSchema;

    fn before_field(&mut self, field: &Field) -> Result<()> {
        self.push_field_context(field, field.name());
        Ok(())
    }

    fn after_field(&mut self, _field: &Field) -> Result<()> {
        self.pop_field_context();
        Ok(())
    }

    fn before_list_element(&mut self, field: &Field) -> Result<()> {
        // Use normalized "element" name for mapping lookup
        self.push_field_context(field, LIST_ELEMENT_NAME);
        Ok(())
    }

    fn after_list_element(&mut self, _field: &Field) -> Result<()> {
        self.pop_field_context();
        Ok(())
    }

    fn before_map_key(&mut self, field: &Field) -> Result<()> {
        // Use normalized "key" name for mapping lookup
        self.push_field_context(field, MAP_KEY_NAME);
        Ok(())
    }

    fn after_map_key(&mut self, _field: &Field) -> Result<()> {
        self.pop_field_context();
        Ok(())
    }

    fn before_map_value(&mut self, field: &Field) -> Result<()> {
        // Use normalized "value" name for mapping lookup
        self.push_field_context(field, MAP_VALUE_NAME);
        Ok(())
    }

    fn after_map_value(&mut self, _field: &Field) -> Result<()> {
        self.pop_field_context();
        Ok(())
    }

    fn schema(&mut self, schema: &ArrowSchema, values: Vec<Self::T>) -> Result<Self::U> {
        Ok(ArrowSchema::new_with_metadata(
            values,
            schema.metadata().clone(),
        ))
    }

    fn r#struct(&mut self, _fields: &Fields, results: Vec<Self::T>) -> Result<Self::T> {
        let info = self
            .current_field_info()
            .expect("Field info stack should not be empty in struct");
        let struct_type = DataType::Struct(results.into());
        let field =
            Field::new(&info.name, struct_type, info.nullable).with_metadata(info.metadata.clone());
        Ok(self.apply_field_id(field, &info.lookup_name))
    }

    fn list(&mut self, list: &DataType, value: Self::T) -> Result<Self::T> {
        let info = self
            .current_field_info()
            .expect("Field info stack should not be empty in list");

        // Apply field ID to the element field
        let element_with_id = self.apply_field_id(value, LIST_ELEMENT_NAME);

        let list_type = match list {
            DataType::List(_) => DataType::List(Arc::new(element_with_id)),
            DataType::LargeList(_) => DataType::LargeList(Arc::new(element_with_id)),
            DataType::FixedSizeList(_, size) => {
                DataType::FixedSizeList(Arc::new(element_with_id), *size)
            }
            _ => unreachable!("list() called with non-list type"),
        };

        let field =
            Field::new(&info.name, list_type, info.nullable).with_metadata(info.metadata.clone());
        Ok(self.apply_field_id(field, &info.lookup_name))
    }

    fn map(&mut self, map: &DataType, key_value: Self::T, value: Self::T) -> Result<Self::T> {
        let info = self
            .current_field_info()
            .expect("Field info stack should not be empty in map");

        // Apply field IDs to key and value fields
        let key_with_id = self.apply_field_id(key_value, MAP_KEY_NAME);
        let value_with_id = self.apply_field_id(value, MAP_VALUE_NAME);

        let (sorted, entries_name) = match map {
            DataType::Map(entries, sorted) => (*sorted, entries.name().clone()),
            _ => unreachable!("map() called with non-map type"),
        };

        let entries_field = Field::new(
            &entries_name,
            DataType::Struct(vec![key_with_id, value_with_id].into()),
            false,
        );

        let map_type = DataType::Map(Arc::new(entries_field), sorted);
        let field =
            Field::new(&info.name, map_type, info.nullable).with_metadata(info.metadata.clone());
        Ok(self.apply_field_id(field, &info.lookup_name))
    }

    fn primitive(&mut self, p: &DataType) -> Result<Self::T> {
        let info = self
            .current_field_info()
            .expect("Field info stack should not be empty in primitive");
        let field =
            Field::new(&info.name, p.clone(), info.nullable).with_metadata(info.metadata.clone());
        Ok(self.apply_field_id(field, &info.lookup_name))
    }
}

/// A wrapper to convert root-level MappedFields to Arc for uniform handling.
struct RootMappingAdapter {
    root_fields: Vec<Arc<MappedField>>,
}

impl RootMappingAdapter {
    fn new(name_mapping: &NameMapping) -> Self {
        let root_fields: Vec<Arc<MappedField>> = name_mapping
            .fields()
            .iter()
            .map(|f| Arc::new(f.clone()))
            .collect();
        Self { root_fields }
    }

    fn fields(&self) -> &[Arc<MappedField>] {
        &self.root_fields
    }
}

/// Apply a name mapping to an Arrow schema to assign Iceberg field IDs.
///
/// This function uses `ArrowSchemaVisitor` to traverse the Arrow schema and look up
/// each field name in the provided `NameMapping`. When a match is found, the
/// field ID is added to the field's metadata under `PARQUET_FIELD_ID_META_KEY`.
///
/// Nested types (struct, list, map) are handled by maintaining a stack of mapping
/// contexts during traversal:
/// - Struct fields: looked up by their actual field name
/// - List elements: looked up using the normalized name "element"
/// - Map keys: looked up using the normalized name "key"
/// - Map values: looked up using the normalized name "value"
///
/// # Arguments
///
/// * `arrow_schema` - The Arrow schema to transform
/// * `name_mapping` - The Iceberg name mapping containing field name to ID mappings
///
/// # Returns
///
/// A new `ArrowSchema` with field IDs assigned from the name mapping.
///
/// # Example
///
/// ```ignore
/// use iceberg::arrow::apply_name_mapping;
/// use iceberg::spec::{MappedField, NameMapping};
///
/// let name_mapping = NameMapping::new(vec![
///     MappedField::new(Some(1), vec!["id".to_string()], vec![]),
///     MappedField::new(Some(2), vec!["name".to_string()], vec![]),
///     MappedField::new(Some(3), vec!["location".to_string()], vec![
///         MappedField::new(Some(4), vec!["lat".to_string()], vec![]),
///         MappedField::new(Some(5), vec!["long".to_string()], vec![]),
///     ]),
/// ]);
///
/// let arrow_schema = ArrowSchema::new(vec![...]);
/// let schema_with_ids = apply_name_mapping(&arrow_schema, &name_mapping)?;
/// ```
pub(crate) fn apply_name_mapping(
    arrow_schema: &ArrowSchema,
    name_mapping: &NameMapping,
) -> Result<ArrowSchema> {
    let adapter = RootMappingAdapter::new(name_mapping);
    let mut visitor = ApplyNameMappingVisitor::new(adapter.fields());
    visit_schema(arrow_schema, &mut visitor)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

    use super::*;
    use crate::spec::MappedField;

    fn get_field_id(field: &Field) -> Option<i32> {
        field
            .metadata()
            .get(PARQUET_FIELD_ID_META_KEY)
            .and_then(|v| v.parse().ok())
    }

    #[test]
    fn test_apply_name_mapping_simple() {
        let name_mapping = NameMapping::new(vec![
            MappedField::new(Some(1), vec!["id".to_string()], vec![]),
            MappedField::new(Some(2), vec!["name".to_string()], vec![]),
            MappedField::new(Some(3), vec!["value".to_string()], vec![]),
        ]);

        let arrow_schema = ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("value", DataType::Float64, true),
        ]);

        let result = apply_name_mapping(&arrow_schema, &name_mapping).unwrap();

        assert_eq!(get_field_id(result.field(0)), Some(1));
        assert_eq!(get_field_id(result.field(1)), Some(2));
        assert_eq!(get_field_id(result.field(2)), Some(3));
    }

    #[test]
    fn test_apply_name_mapping_with_aliases() {
        let name_mapping = NameMapping::new(vec![MappedField::new(
            Some(1),
            vec!["id".to_string(), "record_id".to_string()],
            vec![],
        )]);

        let arrow_schema = ArrowSchema::new(vec![Field::new("record_id", DataType::Int64, false)]);

        let result = apply_name_mapping(&arrow_schema, &name_mapping).unwrap();

        assert_eq!(get_field_id(result.field(0)), Some(1));
    }

    #[test]
    fn test_apply_name_mapping_nested_struct() {
        let name_mapping = NameMapping::new(vec![
            MappedField::new(Some(1), vec!["id".to_string()], vec![]),
            MappedField::new(Some(2), vec!["location".to_string()], vec![
                MappedField::new(
                    Some(3),
                    vec!["latitude".to_string(), "lat".to_string()],
                    vec![],
                ),
                MappedField::new(
                    Some(4),
                    vec!["longitude".to_string(), "long".to_string()],
                    vec![],
                ),
            ]),
        ]);

        let arrow_schema = ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new(
                "location",
                DataType::Struct(
                    vec![
                        Field::new("lat", DataType::Float64, true),
                        Field::new("long", DataType::Float64, true),
                    ]
                    .into(),
                ),
                true,
            ),
        ]);

        let result = apply_name_mapping(&arrow_schema, &name_mapping).unwrap();

        assert_eq!(get_field_id(result.field(0)), Some(1));
        assert_eq!(get_field_id(result.field(1)), Some(2));

        if let DataType::Struct(fields) = result.field(1).data_type() {
            assert_eq!(get_field_id(&fields[0]), Some(3));
            assert_eq!(get_field_id(&fields[1]), Some(4));
        } else {
            panic!("Expected struct type");
        }
    }

    #[test]
    fn test_apply_name_mapping_list() {
        let name_mapping = NameMapping::new(vec![MappedField::new(
            Some(1),
            vec!["items".to_string()],
            vec![MappedField::new(
                Some(2),
                vec!["element".to_string()],
                vec![],
            )],
        )]);

        let arrow_schema = ArrowSchema::new(vec![Field::new(
            "items",
            DataType::List(Arc::new(Field::new("element", DataType::Int32, true))),
            true,
        )]);

        let result = apply_name_mapping(&arrow_schema, &name_mapping).unwrap();

        assert_eq!(get_field_id(result.field(0)), Some(1));

        if let DataType::List(element_field) = result.field(0).data_type() {
            assert_eq!(get_field_id(element_field.as_ref()), Some(2));
        } else {
            panic!("Expected list type");
        }
    }

    #[test]
    fn test_apply_name_mapping_map() {
        let name_mapping = NameMapping::new(vec![MappedField::new(
            Some(1),
            vec!["properties".to_string()],
            vec![
                MappedField::new(Some(2), vec!["key".to_string()], vec![]),
                MappedField::new(Some(3), vec!["value".to_string()], vec![]),
            ],
        )]);

        let entries_field = Field::new(
            "entries",
            DataType::Struct(
                vec![
                    Field::new("key", DataType::Utf8, false),
                    Field::new("value", DataType::Int32, true),
                ]
                .into(),
            ),
            false,
        );

        let arrow_schema = ArrowSchema::new(vec![Field::new(
            "properties",
            DataType::Map(Arc::new(entries_field), false),
            true,
        )]);

        let result = apply_name_mapping(&arrow_schema, &name_mapping).unwrap();

        assert_eq!(get_field_id(result.field(0)), Some(1));

        if let DataType::Map(entries, _) = result.field(0).data_type() {
            if let DataType::Struct(kv_fields) = entries.data_type() {
                assert_eq!(get_field_id(&kv_fields[0]), Some(2));
                assert_eq!(get_field_id(&kv_fields[1]), Some(3));
            } else {
                panic!("Expected struct in map entries");
            }
        } else {
            panic!("Expected map type");
        }
    }

    #[test]
    fn test_apply_name_mapping_missing_field() {
        let name_mapping = NameMapping::new(vec![
            MappedField::new(Some(1), vec!["id".to_string()], vec![]),
            // "name" is not in the mapping
        ]);

        let arrow_schema = ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]);

        let result = apply_name_mapping(&arrow_schema, &name_mapping).unwrap();

        assert_eq!(get_field_id(result.field(0)), Some(1));
        assert_eq!(get_field_id(result.field(1)), None); // No mapping for "name"
    }

    #[test]
    fn test_apply_name_mapping_deeply_nested() {
        // Test deeply nested struct: a -> b -> c -> d
        let name_mapping = NameMapping::new(vec![MappedField::new(
            Some(1),
            vec!["a".to_string()],
            vec![MappedField::new(Some(2), vec!["b".to_string()], vec![
                MappedField::new(Some(3), vec!["c".to_string()], vec![MappedField::new(
                    Some(4),
                    vec!["d".to_string()],
                    vec![],
                )]),
            ])],
        )]);

        let arrow_schema = ArrowSchema::new(vec![Field::new(
            "a",
            DataType::Struct(
                vec![Field::new(
                    "b",
                    DataType::Struct(
                        vec![Field::new(
                            "c",
                            DataType::Struct(vec![Field::new("d", DataType::Int32, true)].into()),
                            true,
                        )]
                        .into(),
                    ),
                    true,
                )]
                .into(),
            ),
            true,
        )]);

        let result = apply_name_mapping(&arrow_schema, &name_mapping).unwrap();

        assert_eq!(get_field_id(result.field(0)), Some(1));

        // Navigate to nested fields
        if let DataType::Struct(b_fields) = result.field(0).data_type() {
            assert_eq!(get_field_id(&b_fields[0]), Some(2));
            if let DataType::Struct(c_fields) = b_fields[0].data_type() {
                assert_eq!(get_field_id(&c_fields[0]), Some(3));
                if let DataType::Struct(d_fields) = c_fields[0].data_type() {
                    assert_eq!(get_field_id(&d_fields[0]), Some(4));
                } else {
                    panic!("Expected struct at level d");
                }
            } else {
                panic!("Expected struct at level c");
            }
        } else {
            panic!("Expected struct at level b");
        }
    }

    #[test]
    fn test_apply_name_mapping_list_of_structs() {
        let name_mapping = NameMapping::new(vec![MappedField::new(
            Some(1),
            vec!["locations".to_string()],
            vec![MappedField::new(
                Some(2),
                vec!["element".to_string()],
                vec![
                    MappedField::new(Some(3), vec!["lat".to_string()], vec![]),
                    MappedField::new(Some(4), vec!["long".to_string()], vec![]),
                ],
            )],
        )]);

        let element_struct = DataType::Struct(
            vec![
                Field::new("lat", DataType::Float64, true),
                Field::new("long", DataType::Float64, true),
            ]
            .into(),
        );

        let arrow_schema = ArrowSchema::new(vec![Field::new(
            "locations",
            DataType::List(Arc::new(Field::new("element", element_struct, true))),
            true,
        )]);

        let result = apply_name_mapping(&arrow_schema, &name_mapping).unwrap();

        assert_eq!(get_field_id(result.field(0)), Some(1));

        if let DataType::List(element_field) = result.field(0).data_type() {
            assert_eq!(get_field_id(element_field.as_ref()), Some(2));
            if let DataType::Struct(struct_fields) = element_field.data_type() {
                assert_eq!(get_field_id(&struct_fields[0]), Some(3));
                assert_eq!(get_field_id(&struct_fields[1]), Some(4));
            } else {
                panic!("Expected struct in list element");
            }
        } else {
            panic!("Expected list type");
        }
    }

    #[test]
    fn test_apply_name_mapping_map_with_nested_value() {
        // Map<String, Struct{lat, long}>
        let name_mapping = NameMapping::new(vec![MappedField::new(
            Some(1),
            vec!["locations_map".to_string()],
            vec![
                MappedField::new(Some(2), vec!["key".to_string()], vec![]),
                MappedField::new(Some(3), vec!["value".to_string()], vec![
                    MappedField::new(Some(4), vec!["lat".to_string()], vec![]),
                    MappedField::new(Some(5), vec!["long".to_string()], vec![]),
                ]),
            ],
        )]);

        let value_struct = DataType::Struct(
            vec![
                Field::new("lat", DataType::Float64, true),
                Field::new("long", DataType::Float64, true),
            ]
            .into(),
        );

        let entries_field = Field::new(
            "entries",
            DataType::Struct(
                vec![
                    Field::new("key", DataType::Utf8, false),
                    Field::new("value", value_struct, true),
                ]
                .into(),
            ),
            false,
        );

        let arrow_schema = ArrowSchema::new(vec![Field::new(
            "locations_map",
            DataType::Map(Arc::new(entries_field), false),
            true,
        )]);

        let result = apply_name_mapping(&arrow_schema, &name_mapping).unwrap();

        assert_eq!(get_field_id(result.field(0)), Some(1));

        if let DataType::Map(entries, _) = result.field(0).data_type() {
            if let DataType::Struct(kv_fields) = entries.data_type() {
                assert_eq!(get_field_id(&kv_fields[0]), Some(2)); // key
                assert_eq!(get_field_id(&kv_fields[1]), Some(3)); // value

                // Check nested struct in value
                if let DataType::Struct(value_fields) = kv_fields[1].data_type() {
                    assert_eq!(get_field_id(&value_fields[0]), Some(4)); // lat
                    assert_eq!(get_field_id(&value_fields[1]), Some(5)); // long
                } else {
                    panic!("Expected struct in map value");
                }
            } else {
                panic!("Expected struct in map entries");
            }
        } else {
            panic!("Expected map type");
        }
    }
}
