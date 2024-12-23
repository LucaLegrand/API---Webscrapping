import google.auth
from google.cloud import firestore

class FirestoreClient:

    client: firestore.Client

    def __init__(self, database_id: str = "(default)") -> None:
        credentials, _ = google.auth.default()
        self.client = firestore.Client(
            credentials=credentials, project=credentials.project_id, database=database_id)

    def get(self, collection_name: str, document_id: str) -> dict:
        doc = self.client.collection(
            collection_name).document(document_id).get()
        if doc.exists:
            return doc.to_dict()
        raise FileNotFoundError(
            f"No document found at {collection_name} with the ID {document_id}"
        )

    def get_parameters(self) -> dict:
        return self.get(collection_name="parameters", document_id="parameters")

    def update_or_add_parameters(self, parameters: dict) -> dict:
        collection_name = "parameters"
        document_id = "parameters"
        doc_ref = self.client.collection(collection_name).document(document_id)
        doc_ref.set(parameters, merge=True)
        return parameters

