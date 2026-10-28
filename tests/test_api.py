
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

SAMPLE_STUDENT = {
    "school": "GP",
    "sex": "F",
    "age": 16,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": 4,
    "Fedu": 3,
    "Mjob": "teacher",
    "Fjob": "other",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 1,
    "studytime": 3,
    "failures": 0,
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 2,
    "Dalc": 1,
    "Walc": 2,
    "health": 4,
    "absences": 2,
    "G1": 12,
    "G2": 13,
}


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Student Performance API is running"
    )


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint():
    response = client.post("/predict", json=SAMPLE_STUDENT)

    assert response.status_code == 200

    result = response.json()

    assert "predicted_G3" in result
    assert isinstance(result["predicted_G3"], (int, float))
    assert 0 <= result["predicted_G3"] <= 20
    assert result["scale"] == "0-20"


def test_invalid_age_is_rejected():
    student = SAMPLE_STUDENT.copy()
    student["age"] = 100

    response = client.post("/predict", json=student)

    assert response.status_code == 422


def test_missing_feature_is_rejected():
    student = SAMPLE_STUDENT.copy()
    del student["G1"]

    response = client.post("/predict", json=student)

    assert response.status_code == 422
