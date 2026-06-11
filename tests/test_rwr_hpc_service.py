import threading
import unittest

from runtime.rwr_hpc_service import RwrHpcServiceClient, RwrHpcServiceState, build_rwr_hpc_server
from runtime.schemas import ToolAction, ToolObservation, ToolObservationStatus


class RwrHpcServiceTests(unittest.TestCase):
    def test_client_server_round_trip_records_metrics(self) -> None:
        def fake_executor(action: ToolAction) -> ToolObservation:
            return ToolObservation(
                status=ToolObservationStatus.SUCCESS,
                provenance={"tool_name": action.tool_name, "cache_hit": True},
                call_id=action.call_id,
                payload={"tool_name": action.tool_name, "ok": True},
            )

        state = RwrHpcServiceState(fake_executor)
        server = build_rwr_hpc_server(host="127.0.0.1", port=0, state=state)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            client = RwrHpcServiceClient(f"http://{host}:{port}", timeout_seconds=5)
            result = client.run_tool(
                ToolAction(
                    tool_name="rwr",
                    arguments={"seed_genes": ["ENSG1"], "top_k": 1},
                    call_id="call-1",
                )
            )

            self.assertFalse(result.is_empty)
            self.assertEqual(result.payload["tool_name"], "rwr")
            self.assertEqual(result.provenance["cache_hit"], True)
            self.assertIn("rwr_hpc_service_url", result.provenance)
            snapshot = state.metrics.snapshot()
            self.assertEqual(snapshot["total_request_count"], 1)
            self.assertEqual(snapshot["total_error_count"], 0)
            self.assertEqual(snapshot["tools"]["rwr"]["cache_hit_rate"], 1.0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
