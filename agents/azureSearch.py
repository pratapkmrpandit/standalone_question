import logging
import os
import traceback
import openai
from azure.search.documents.indexes.models import (
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
)

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from standalone import settings



class AzureSearchAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.open_ai_error_code = None
        self.index_name = settings.vdb_conf["collection_name"]
        self.component_index = settings.vdb_conf["component_index"]
        self.embeddings = AzureOpenAIEmbeddings(
            model=settings.config['embedding_model'],
            api_version=os.getenv("EMBEDDING_API_VERSION"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
        if self.component_index:
            self.component_index_vector_store, self.open_ai_error_code = (
                self.get_component_index()
            )
        if self.index_name:
            self.vector_store, self.open_ai_error_code = (
                self.get_azure_index()
            )

    def get_component_index(self):
        """
        Initialize AzureSearch vector store for the component index safely.
        Avoids making API calls during field definition.
        """
        open_ai_error_code = None
        vector_store = None

        try:
            # Use fixed embedding dimension based on your model
            MODEL_DIMENSIONS = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
            }
            embedding_model = settings.config['embedding_model']
            embedding_dim = MODEL_DIMENSIONS.get(embedding_model, 1536)  # default to 1536

            # Define fields without calling embed_query
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
                SearchableField(name="content", type=SearchFieldDataType.String, searchable=True),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=embedding_dim,
                    vector_search_profile_name="myHnswProfile",
                ),
                SearchableField(name="metadata", type=SearchFieldDataType.String, searchable=True),
                SearchableField(name="title", type=SearchFieldDataType.String, searchable=True),
                SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, searchable=True),
                SimpleField(name="modulename", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="componentname", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="tablename", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="sqlquery", type=SearchFieldDataType.String, filterable=True)
            ]
            print(fields)
            # Initialize AzureSearch vector store
            vector_store = AzureSearch(
                azure_search_endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
                azure_search_key=os.getenv("AZURE_AI_SEARCH_KEY"),
                index_name=self.component_index,
                embedding_function=self.embeddings.embed_query,  # only used at query time
                semantic_configuration_name="default",
                fields=fields,
            )

            self.logger.info(f"Component index vector store initialized: {vector_store}")

        except openai.OpenAIError as e:
            if "quota" in str(e):
                open_ai_error_code = "429_quota"
            elif "rate" in str(e):
                open_ai_error_code = "429_rate_limit"
            else:
                open_ai_error_code = str(e.status_code)
            self.logger.error(f"OpenAI error during vector store init: {str(e)}")

        except Exception:
            open_ai_error_code = "generic_error"
            self.logger.error(
                "Something went wrong in get_component_index :: " + traceback.format_exc()
            )
        return vector_store, open_ai_error_code

    def get_azure_index(self):
        vector_store = None
        open_ai_error_code = None
        try:

            fields = [
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True,
                    filterable=True,
                ),
                SearchableField(
                    name="content",
                    type=SearchFieldDataType.String,
                    searchable=True,
                ),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=len(self.embeddings.embed_query("Text")),
                    vector_search_profile_name="myHnswProfile",
                ),
                SearchableField(
                    name="metadata",
                    type=SearchFieldDataType.String,
                    searchable=True,
                ),
                # Additional field to store the title
                SearchableField(
                    name="title",
                    type=SearchFieldDataType.String,
                    searchable=True,
                ),
                # Additional field for filtering on document source
                SimpleField(
                    name="source",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    searchable=True,
                ),
                SimpleField(
                    name="componentname",
                    type=SearchFieldDataType.String,
                    filterable=True,
                ),
                SimpleField(
                    name="tablename",
                    type=SearchFieldDataType.String,
                    filterable=True,
                ),
                SimpleField(
                    name="modulename",
                    type=SearchFieldDataType.String,
                    filterable=True,
                ),
                SimpleField(
                    name="headerdetail",
                    type=SearchFieldDataType.String,
                    filterable=True,
                ),
            ]

            vector_store = AzureSearch(
                azure_search_endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
                azure_search_key=os.getenv("AZURE_AI_SEARCH_KEY"),
                index_name=self.index_name,
                embedding_function=self.embeddings.embed_query,
                semantic_configuration_name="default",
                fields=fields,
            )

        except openai.OpenAIError as e:
            if "quota" in str(e):
                open_ai_error_code = "429_quota"
            elif "rate" in str(e):
                open_ai_error_code = "429_rate_limit"
            else:
                open_ai_error_code = str(e.status_code)

            self.logger.error("OpenAI error: " + str(e))
        except Exception:
            error = traceback.format_exc()
            self.logger.error(
                "something went wrong in get azure index function :: "
                + str(traceback.format_exc())
            )
        return vector_store, open_ai_error_code