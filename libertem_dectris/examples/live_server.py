import time
import json
import asyncio
import typing
import copy
from collections import OrderedDict
import math

import numba
import numpy as np
import uuid
import websockets
from websockets.legacy.server import WebSocketServerProtocol
from websockets.legacy.client import WebSocketClientProtocol
import bitshuffle

from libertem import masks
from libertem.udf.sum import SumUDF
from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.masks import ApplyMasksUDF
from libertem.udf.com import COMParams
from libertem_icom.udf.icom import ICOMUDF
from libertem.executor.pipelined import PipelinedExecutor
from libertem_live.api import LiveContext
from libertem_live.udf.monitor import (
    SignalMonitorUDF, PartitionMonitorUDF
)

from libertem.udf.base import UDFResults, UDF
from libertem.common.async_utils import sync_to_async

if typing.TYPE_CHECKING:
    from libertem.common.executor import JobExecutor


class EncodedResult:
    def __init__(
            self,
            compressed_data: memoryview,
            bbox: typing.Tuple[int, int, int, int],
            full_shape: typing.Tuple[int, int],
            delta_shape: typing.Tuple[int, int],
            dtype: str,
            channel_name: str,
            udf_name: str,
        ):
        self.compressed_data = compressed_data
        self.bbox = bbox
        self.full_shape = full_shape
        self.delta_shape = delta_shape
        self.dtype = dtype
        self.channel_name = channel_name
        self.udf_name = udf_name

    def is_empty(self):
        return len(self.compressed_data) == 0


@numba.njit(cache=True)
def get_bbox(arr) -> typing.Tuple[int, ...]:
    xmin = arr.shape[1]
    ymin = arr.shape[0]
    xmax = 0
    ymax = 0

    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            value = arr[y, x]
            if abs(value) < 1e-8:
                continue
            # got a non-zero value, update indices
            if x < xmin:
                xmin = x
            if x > xmax:
                xmax = x
            if y < ymin:
                ymin = y
            if y > ymax:
                ymax = y
    return int(ymin), int(ymax), int(xmin), int(xmax)


class SingleMaskUDF(ApplyMasksUDF):
    def get_result_buffers(self):
        dtype = np.result_type(self.meta.input_dtype, self.get_mask_dtype())
        return {
            'intensity': self.buffer(
                kind='nav', extra_shape=(1,), dtype=dtype, where='device', use='internal',
            ),
            'intensity_nav': self.buffer(
                kind='nav', extra_shape=(), dtype=dtype, where='device', use='result',
            ),
        }

    def get_results(self):
        # bummer: we can't reshape the data, as the extra_shape from the buffer
        # will override our desired shape. so we have to use two result buffers
        # instead:
        return {
            'intensity_nav': self.results.intensity.reshape(self.meta.dataset_shape.nav),
        }


class ResultTracker:
    def __init__(self):
        self._data = {}

    def update_result(self, key, result):
        self._data[key] = result

    def get_result(self, key):
        return self._data[key]

    def remove_result(self, key):
        del self._data[key]

    def clear(self):
        self._data = {}

    def keys(self):
        return list(self._data.keys())


class WSServer:
    def __init__(self):
        self.connect()
        self.ws_connected = set()
        self.parameters = {
            'cx': 516/2.0,
            'cy': 512/2.0,
            'ri': 200.0,
            'ro': 530.0,
        }
        self.udfs = self.get_udfs()
        self.results = ResultTracker()
        self.lock = asyncio.Lock()

    def get_udfs(self):
        cx = self.parameters['cx']
        cy = self.parameters['cy']
        ri = self.parameters['ri']
        ro = self.parameters['ro']

        def _ring():
            return masks.ring(
                centerX=cx,
                centerY=cy,
                imageSizeX=516,
                imageSizeY=516,
                radius=ro,
                radius_inner=ri)

        mask_udf = SingleMaskUDF(mask_factories=[_ring])
        params = COMParams(cy=cy, cx=cx, r=ro, ri=ri)
        icom_udf = ICOMUDF(params)
        return OrderedDict({
            # "brightfield": SumSigUDF(),
            # "annular": mask_udf,
            "icom": icom_udf,
            # "annular1": mask_udf,
            # "annular2": mask_udf,
            # "annular3": mask_udf,
            # "annular4": mask_udf,
            # "sum": SumUDF(),
            # "monitor": SignalMonitorUDF(),
            "monitor_partition": PartitionMonitorUDF(),
        })

    async def __call__(self, websocket: WebSocketServerProtocol):
        await self.client_loop(websocket)

    async def send_state_dump(self, websocket: WebSocketClientProtocol):
        await websocket.send(json.dumps({
            'event': 'UPDATE_PARAMS',
            'parameters': self.parameters,
        }))
        for key in self.results.keys():
            result = self.results.get_result(key)
            await self.handle_partial_result(
                result, key, previous_results=None, dest=websocket
            )

    def register_client(self, websocket):
        self.ws_connected.add(websocket)

    def unregister_client(self, websocket):
        self.ws_connected.remove(websocket)

    async def client_loop(self, websocket: WebSocketClientProtocol):
        try:
            try:
                async with self.lock:
                    await self.send_state_dump(websocket)
                    self.register_client(websocket)
                async for msg in websocket:
                    await self.handle_message(msg, websocket)
            except websockets.exceptions.ConnectionClosedError:
                await websocket.close()
        finally:
            self.unregister_client(websocket)

    async def handle_message(self, msg, websocket):
        try:
            msg = json.loads(msg)
            # FIXME: hack to not require the 'event' "tag":
            if 'event' not in msg or msg['event'] == 'UPDATE_PARAMS':
                print(f"parameter update: {msg}")
                self.parameters = msg['parameters']
                self.udfs = self.get_udfs()
                # broadcast to all clients:
                msg['event'] = 'UPDATE_PARAMS'
                await self.broadcast(json.dumps(msg))
        except Exception as e:
            print(e)

    async def broadcast(self, msg):
        websockets.broadcast(self.ws_connected, msg)

    async def make_deltas(self, partial_results: UDFResults, previous_results: typing.Optional[UDFResults]) -> np.ndarray:
        deltas = []
        udf_names = list(self.udfs.keys())
        for idx in range(len(partial_results.buffers)):
            udf_name = udf_names[idx]
            for channel_name in partial_results.buffers[idx].keys():
                data = partial_results.buffers[idx][channel_name].data
                # FIXME implement n-dimnsional result buffers
                if len(data.shape) != 2:
                    continue
                if previous_results is None:
                    data_previous = np.zeros_like(data)
                else:
                    data_previous = previous_results.buffers[idx][channel_name].data

                delta = data - data_previous
                deltas.append({
                    'delta': delta,
                    'udf_name': udf_name,
                    'channel_name': channel_name,
                })
        return deltas

    async def encode_result(self, delta: np.ndarray, udf_name: str, channel_name: str) -> EncodedResult:
        """
        Slice `delta` to its non-zero region and compress that. Returns the information
        needed to reconstruct the the full result.
        """
        nonzero_mask = ~np.isclose(0, delta)

        if np.count_nonzero(nonzero_mask) == 0:
            print(f"zero-delta update, skipping")
            # skip this update if it is all-zero
            return EncodedResult(
                compressed_data=memoryview(b""),
                bbox=(0, 0, 0, 0),
                full_shape=delta.shape,
                delta_shape=(0, 0),
                dtype=delta.dtype,
                channel_name=channel_name,
                udf_name=udf_name,
            )

        bbox = get_bbox(delta)
        ymin, ymax, xmin, xmax = bbox
        delta_for_blit = delta[ymin:ymax + 1, xmin:xmax + 1]
        # print(delta_for_blit.shape, delta_for_blit, list(delta_for_blit[-1]))

        # FIXME: remove allocating copy - maybe copy into pre-allocated buffer instead?
        # compressed = await sync_to_async(lambda: lz4.frame.compress(np.copy(delta_for_blit)))
        compressed = await sync_to_async(lambda: bitshuffle.compress_lz4(np.copy(delta_for_blit)))

        return EncodedResult(
            compressed_data=memoryview(compressed),
            bbox=bbox,
            full_shape=delta.shape,
            delta_shape=delta_for_blit.shape,
            dtype=delta_for_blit.dtype,
            channel_name=channel_name,
            udf_name=udf_name,
        )

    async def handle_pending_acquisition(self, pending) -> str:
        acq_id = str(uuid.uuid4())
        await self.broadcast(json.dumps({
            "event": "ACQUISITION_STARTED",
            "id": acq_id,
        }))
        return acq_id

    async def handle_acquisition_end(self, pending, acq_id: str):
        await self.broadcast(json.dumps({
            "event": "ACQUISITION_STARTED",
            "id": acq_id,
        }))

    async def handle_acquisition_end(self, pending, acq_id: str):
        await self.broadcast(json.dumps({
            "event": "ACQUISITION_ENDED",
            "id": acq_id,
        }))

    async def handle_partial_result(
        self,
        partial_results: UDFResults,
        acq_id: str,
        previous_results: typing.Optional[UDFResults],
        dest: typing.Optional[WebSocketClientProtocol] = None,
    ):
        deltas = await self.make_deltas(partial_results, previous_results)

        delta_results: typing.List[EncodedResult] = []
        for delta in deltas:
            delta_results.append(
                await self.encode_result(
                    delta['delta'],
                    delta['udf_name'],
                    delta['channel_name']
                )
            )
        header_msg = json.dumps({
            "event": "RESULT",
            "id": acq_id,
            "timestamp": time.time(),
            "channels": [
                {
                    "bbox": result.bbox,
                    "full_shape": result.full_shape,
                    "delta_shape": result.delta_shape,
                    "dtype": str(result.dtype),
                    "encoding": "bslz4",
                    "channel_name": result.channel_name,
                    "udf_name": result.udf_name,
                }
                for result in delta_results
            ],
        })
        if dest is None:
            fn = self.broadcast
        else:
            fn = dest.send

        await fn(header_msg)
        for result in delta_results:
            await fn(result.compressed_data)

    async def acquisition_loop(self):
        min_delta = 0.05
        while True:
            pending_aq = await sync_to_async(self.conn.wait_for_acquisition, timeout=10)
            if pending_aq is None:
                continue
            try:
                acq_id = await self.handle_pending_acquisition(pending_aq)
                print(f"acquisition starting with id={acq_id}")
                t0 = time.perf_counter()
                num_updates = 0
                previous_results = None
                partial_results = None
                side = int(math.sqrt(pending_aq.detector_config.get_num_frames()))
                aq = self.ctx.make_acquisition(
                    conn=self.conn,
                    pending_aq=pending_aq,
                    frames_per_partition=side,
                    nav_shape=(side, side),
                )
                last_update = 0
                try:
                    udfs_only = list(self.udfs.values())
                    async for partial_results in self.ctx.run_udf_iter(dataset=aq, udf=udfs_only, sync=False):
                        # XXX hacks: parameter update
                        udfs_only = list(self.udfs.values())
                        if self.ctx._runner._params._kwargs != [udf._kwargs for udf in udfs_only]:
                            key = self.ctx._runner._params_handle
                            self.ctx._runner._params.update_from_udfs(udfs_only)
                            self.ctx.executor.scatter_update(key, self.ctx._runner._params)

                        if time.time() - last_update > min_delta:
                            async with self.lock:
                                await self.handle_partial_result(partial_results, acq_id, previous_results)
                                self.results.update_result(acq_id, partial_results)
                            previous_results = copy.deepcopy(partial_results)
                            last_update = time.time()
                            num_updates += 1
                    async with self.lock:
                        await self.handle_partial_result(partial_results, acq_id, previous_results)
                        self.results.update_result(acq_id, partial_results)
                    num_updates += 1
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.ctx.close()
                    self.conn.close()
                    self.connect()
                previous_results = copy.deepcopy(partial_results)
            finally:
                await self.handle_acquisition_end(pending_aq, acq_id)
                self.results.clear()
            previous_results = None
            t1 = time.perf_counter()
            print(f"acquisition done with id={acq_id}; took {t1-t0:.3f}s; num_updates={num_updates}")
            num_updates = 0

    async def serve(self):
        async with websockets.serve(self, "localhost", 8444):
            try:
                await self.acquisition_loop()
            finally:
                self.conn.close()
                self.ctx.close()

    def connect(self):
        executor = PipelinedExecutor(
            spec=PipelinedExecutor.make_spec(
                cpus=range(24), cudas=[]
            ),
            pin_workers=False,
        )
        ctx = LiveContext(executor=executor)
        conn = ctx.make_connection('dectris').open(
            api_host='localhost',
            api_port=8910,
            data_host='localhost',
            data_port=9999,
            buffer_size=2048,
        )

        self.conn = conn
        self.executor = executor
        self.ctx = ctx

async def main():
    server = WSServer()
    await server.serve()


if __name__ == "__main__":

    asyncio.run(main())
