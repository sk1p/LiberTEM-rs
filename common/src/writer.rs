use std::path::PathBuf;

use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use zarrs::array::FillValue;

use crate::zarr_writer::{ZarrCompression, ZarrWriter};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SaveFormat {
    Zarr {
        path: PathBuf,
        array_name: String,
        compression: Option<ZarrCompression>,
    },

    /// hyperspy-compatible zarr file
    ///
    /// # Structure:
    ///
    /// ## The array:
    ///
    ///     /Experiments/<experiment_name>/data
    ///
    /// ## Axes:
    ///
    ///     /Experiments/<experiment_name>/axis-0
    ///     /Experiments/<experiment_name>/axis-...
    ///     /Experiments/<experiment_name>/axis-N
    ///
    /// Where N is the number of dimensions minus one
    ///
    /// ## Metadata:
    ///
    ///     /Experiments/<experiment_name>/metadata/...
    ///
    Zspy {
        path: PathBuf,
        experiment_name: String,
        compression: Option<ZarrCompression>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveOptions {
    format: SaveFormat,
}

impl SaveOptions {
    pub fn new(format: SaveFormat) -> Self {
        Self { format }
    }

    /// Perform any work necessary to initialize the file.
    /// For example with zarr, array metadata is written here.
    pub fn create(&self) {
        // TODO: Result<...>
        // TODO: refactor once we support a non-zarr writer
        match &self.format {
            SaveFormat::Zarr {
                path,
                array_name,
                compression,
            } => {
                let dtype = zarrs::array::DataType::UInt16;
                let fill_value = match dtype {
                    zarrs::array::DataType::Bool => FillValue::from(false),
                    zarrs::array::DataType::Int8 => FillValue::from(0i8),
                    zarrs::array::DataType::Int16 => FillValue::from(0i16),
                    zarrs::array::DataType::Int32 => FillValue::from(0i32),
                    zarrs::array::DataType::Int64 => FillValue::from(0i64),
                    zarrs::array::DataType::UInt8 => FillValue::from(0u8),
                    zarrs::array::DataType::UInt16 => FillValue::from(0u16),
                    zarrs::array::DataType::UInt32 => FillValue::from(0u32),
                    zarrs::array::DataType::UInt64 => FillValue::from(0u64),
                    zarrs::array::DataType::Float16 => FillValue::from(f16::from_f32_const(0f32)),
                    zarrs::array::DataType::Float32 => FillValue::from(0f32),
                    zarrs::array::DataType::Float64 => FillValue::from(0f64),
                    zarrs::array::DataType::BFloat16 => FillValue::from(bf16::from_f32_const(0f32)),
                    zarrs::array::DataType::Complex64 => {
                        FillValue::from(num::complex::Complex32::new(0f32, 0f32))
                    }
                    zarrs::array::DataType::Complex128 => {
                        FillValue::from(num::complex::Complex64::new(0f64, 0f64))
                    }
                    zarrs::array::DataType::RawBits(size) => FillValue::from(&vec![0u8; size][..]),
                    zarrs::array::DataType::String => FillValue::from(String::new()),
                    zarrs::array::DataType::Binary => FillValue::from(&[0u8; 1]),
                    _ => todo!(),
                };
                ZarrWriter::init(
                    path,
                    array_name,
                    compression.as_ref(),
                    chunk_grid,
                    array_shape,
                    dtype,
                    fill_value,
                )
                .unwrap();
            }
            SaveFormat::Zspy {
                path,
                experiment_name,
                compression,
            } => todo!(),
        }
    }
}
