import importlib.util
import unittest


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class ApiTests(unittest.TestCase):
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient

        from api.app import create_app

        client = TestClient(create_app())
        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_service_index_links_demo_endpoints(self):
        from fastapi.testclient import TestClient

        from api.app import create_app

        client = TestClient(create_app())
        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["scope"], "public synthetic-data demo")
        self.assertIn("/rag/search", payload["rag_search_example"])

    def test_compare_request_validates_seed(self):
        from fastapi.testclient import TestClient

        from api.app import create_app

        client = TestClient(create_app())
        response = client.post("/compare-policies", json={"scenario": "default_run", "seed": -1})

        self.assertEqual(response.status_code, 422)

    def test_metrics_endpoint_reports_requests(self):
        from fastapi.testclient import TestClient

        from api.app import create_app

        client = TestClient(create_app())
        client.get("/health")
        response = client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(response.json()["request_count"], 1)

    def test_pipeline_status_endpoint_is_available(self):
        from fastapi.testclient import TestClient

        from api.app import create_app

        response = TestClient(create_app()).get("/pipeline-status")

        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json()["status"], {"succeeded", "failed_quality", "not_run"})


if __name__ == "__main__":
    unittest.main()
