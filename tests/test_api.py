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

    def test_compare_request_validates_seed(self):
        from fastapi.testclient import TestClient

        from api.app import create_app

        client = TestClient(create_app())
        response = client.post("/compare-policies", json={"scenario": "default_run", "seed": -1})

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
