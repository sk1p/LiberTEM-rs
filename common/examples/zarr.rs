//! Prototype for writing to zarr data sets. Just to figure out the API etc.
//!
//! * What happens if the array shape isn't divisible by the chunk shape?
//!   https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#regular-grids
//!   "If the length of any array dimension is not perfectly divisible by the
//!   chunk length along the same dimension, then the grid will overhang the edge
//!   of the array space."
//!
//! * If we have a continuously running acquisition, we can theoretically just
//!   continuously write chunks, and fix the array shape at the end, with the last
//!   chunk being smaller than the others.
//!
//! * As for write performance, the default `FilesystemStore` does use "normal"
//!   buffered I/O. This is an issue in high-throughput situations, as memory
//!   pressure continuously rises and write times are non-deterministic.
//!
//!   The good news is that we can override the implementation, basically forking
//!   `FilesystemStore` (we can't easily just delegate, as that would require access
//!   to some implementation details, for example for locking).
//!
//! * The chunking has to be equal to partition sizes, as we can only have overhang
//!   "at the edge of the array space", meaning we can't use smaller units,
//!   where we may have overhead at the end of every partition. But we can use sharding
//!   to subdivide the partition into individually addressable sub-chunks, which
//!   means we don't have read amplification like we would with HDF5.

use std::sync::Arc;

use zarrs::{
    array::{
        codec::{array_to_bytes::sharding::ShardingCodecBuilder, ZstdCodec},
        ArrayBuilder, FillValue,
    },
    group::GroupBuilder,
    storage::{
        store::{self},
        ReadableWritableListableStorage,
    },
};

pub fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let mut store: ReadableWritableListableStorage =
        Arc::new(store::FilesystemStore::new("/tmp/libertem-zarr-writer/")?);

    let mut group = GroupBuilder::new().build(Arc::clone(&store), "/group")?;
    group.store_metadata()?;

    let shard_shape = vec![1, 3, 16, 31];
    let inner_chunk_shape = vec![1, 1, 16, 31];

    let mut sharding_codec_builder =
        ShardingCodecBuilder::new(inner_chunk_shape.as_slice().try_into()?);
    sharding_codec_builder.bytes_to_bytes_codecs(vec![Box::new(ZstdCodec::new(5, true))]);

    let array = ArrayBuilder::new(
        vec![8, 3, 16, 31],
        zarrs::array::DataType::Float32,
        shard_shape.try_into()?,
        FillValue::from(0.0f32),
    )
    .array_to_bytes_codec(Box::new(sharding_codec_builder.build()))
    .dimension_names(["y", "x", "Ky", "Kx"].into())
    .build(Arc::clone(&store), "/array")?;

    array.store_metadata()?;

    println!(
        "metadata: {}",
        serde_json::to_string_pretty(&array.metadata())?
    );

    // this is what would happen on each worker process, for each partition:
    {
        // let's get another reference to the array:
        let arr2 = zarrs::array::Array::open(Arc::clone(&store), "/array")?;

        // .. and write a chunk:
        let chunk_grid = arr2.chunk_grid();
        let chunk_indices = vec![0, 0, 0, 0];

        let chunk_shape = chunk_grid
            .chunk_shape(&chunk_indices, arr2.shape())?
            .unwrap();

        // at least one element must be non-equal to the `FillValue` defined above;
        // otherwise, by default, the chunk will not be written at all.
        let elements = vec![1.2f32; 3 * 16 * 31];

        println!(
            "storing chunk array {:?} {:?} {:?}",
            &chunk_indices, chunk_shape, chunk_grid
        );
        arr2.store_chunk_elements(&chunk_indices, &elements)?;

        // `store_chunk_ndarray` allocates a temporary `Vec`, which is not something
        // anyone ever would want to have in their data path... ugh. (I can
        // understand this may need to happen if the array is not in C order or
        // non-contiguous... but still, ugh)
        // arr2.store_chunk_ndarray(&chunk_indices, chunk_array)?;
    }

    Ok(())
}
