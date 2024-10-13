import os
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import Settings
from pinecone import Pinecone
from llama_index.core import PromptTemplate
from llama_index.llms.text_generation_inference import TextGenerationInference
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Event
from llama_index.core.tools import QueryEngineTool
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.core import StorageContext
from llama_index.core.schema import NodeWithScore
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.workflow import Workflow, step, Context, StartEvent, StopEvent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
from colorama import Fore, Style
from dotenv import load_dotenv
from typing import Optional, List, Callable
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

load_dotenv()

# Initialize Settings
Settings.llm = OpenAI(model="gpt-4", temperature=0.1, api_key="<api>")
# embed_model = MistralAIEmbedding(model_name="mistral-embed")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5")


### --- Initialize Pinecone and Vector Stores --- ###

pc = Pinecone(api_key="aafc89a7-408c-4bbc-8193-58e103949c98")

# Assume INDEX_NAME is an environment variable for each index
image_captioning_index = pc.Index("llamaindex-ragathon-demo-index-v2")
transcripts_index = pc.Index("llamaindex-ragathon-demo-index-v2")
yolo_index = pc.Index("llamaindex-ragathon-demo-index-v2")

# Create Vector Stores
image_captioning_vector_store = PineconeVectorStore(pinecone_index=image_captioning_index, namespace="test_05_14")
transcripts_vector_store = PineconeVectorStore(pinecone_index=transcripts_index, namespace="test_05_14")
yolo_vector_store = PineconeVectorStore(pinecone_index=yolo_index, namespace="test_05_14")

# Create Storage Contexts
image_captioning_storage_context = StorageContext.from_defaults(
    vector_store=image_captioning_vector_store
)
transcripts_storage_context = StorageContext.from_defaults(
    vector_store=transcripts_vector_store
)
yolo_storage_context = StorageContext.from_defaults(
    vector_store=yolo_vector_store
)

# Create Vector Store Indexes
image_captioning_vector_store_index = VectorStoreIndex.from_vector_store(
    image_captioning_vector_store
)
transcripts_vector_store_index = VectorStoreIndex.from_vector_store(
    transcripts_vector_store
)
yolo_vector_store_index = VectorStoreIndex.from_vector_store(yolo_vector_store)

### --- Prompt Template --- ###

qa_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

### --- Define Custom RAG Query Engine --- ###

class RAGStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: TextGenerationInference
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        print('RAG-custom_query: ',query_str, len(nodes))
        for n in nodes:
            print(n)
        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        response = self.llm.complete(
            self.qa_prompt.format(context_str=context_str, query_str=query_str)
        )
        return str(response)

# Initialize LLM for RAG
# hf_llm = TextGenerationInference(
#     model_name="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
#     model_url=os.environ["HF_LLM_URL"],
#     token=os.environ["HF_TOKEN"],
# )

gpt=TextGenerationInference(
    model_url="https://run-execution-0eie59936uqw-run-execution-8000.oregon.google-cluster.vessl.ai/v1/chat/completions",
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    temperature=0.7,
    max_tokens=100,
)


filters_image_caption = MetadataFilters(
    filters=[
        MetadataFilter(
            key="agent", operator=FilterOperator.EQ, value="image_captioning"
        ),
    ]
)
filters_transcripts = MetadataFilters(
    filters=[
        MetadataFilter(
            key="agent", operator=FilterOperator.EQ, value="transcripts"
        ),
    ]
)
filters_yolo = MetadataFilters(
    filters=[
        MetadataFilter(
            key="agent", operator=FilterOperator.EQ, value="yolo"
        ),
    ]
)
# Create Retrievers
image_captioning_retriever = image_captioning_vector_store_index.as_retriever(filters=filters_image_caption)
transcripts_retriever = transcripts_vector_store_index.as_retriever(filters=filters_transcripts)
yolo_retriever = yolo_vector_store_index.as_retriever(filters=filters_yolo)

# Create Response Synthesizer
synthesizer = get_response_synthesizer(response_mode="compact")

# Create RAG Query Engines for each tool
image_captioning_query_engine = RAGStringQueryEngine(
    retriever=image_captioning_retriever,
    response_synthesizer=synthesizer,
    llm=gpt,
    qa_prompt=qa_prompt,
)

transcripts_query_engine = RAGStringQueryEngine(
    retriever=transcripts_retriever,
    response_synthesizer=synthesizer,
    llm=gpt,
    qa_prompt=qa_prompt,
)

yolo_query_engine = RAGStringQueryEngine(
    retriever=yolo_retriever,
    response_synthesizer=synthesizer,
    llm=gpt,
    qa_prompt=qa_prompt,
)

# Create QueryEngineTools for each RAG Query Engine
image_captioning_tool = QueryEngineTool.from_defaults(
    image_captioning_query_engine,
    name="image_captioning_tool",
    description="Useful for answering questions about the video/image scens and captioning of the images/videos"
)

transcripts_tool = QueryEngineTool.from_defaults(
    transcripts_query_engine,
    name="transcripts_tool",
    description="Useful for answering questions about video transcripts and audio related questions"
)

yolo_tool = QueryEngineTool.from_defaults(
    yolo_query_engine,
    name="yolo_tool",
    description="Useful for answering questions about objects detected in video frames."
)

### --- Define Events --- ###

class InitializeEvent(Event):
    pass

class InterfaceAgentEvent(Event):
    request: Optional[str] = None
    just_completed: Optional[str] = None
    need_help: Optional[bool] = None

class OrchestratorEvent(Event):
    request: str

class ImageCaptioningEvent(Event):
    request: str

class TranscriptsEvent(Event):
    request: str

class YoloEvent(Event):
    request: str

### --- Define BaseAgent --- ###

class BaseAgent():
    name: str
    parent: Workflow
    tools: List[FunctionTool]
    system_prompt: str
    context: Context
    current_event: Event
    trigger_event: Event

    def __init__(
            self,
            parent: Workflow,
            tools: List[Callable],
            system_prompt: str,
            trigger_event: Event,
            context: Context,
            name: str,
        ):
        self.name = name
        self.parent = parent
        self.context = context
        self.system_prompt = system_prompt
        self.context.data["redirecting"] = False
        self.trigger_event = trigger_event

        # Set up the tools including the ones everybody gets
        def done() -> None:
            """When you complete your task, call this tool."""
            print(f"{self.name} is complete")
            self.context.data["redirecting"] = True
            parent.send_event(InterfaceAgentEvent(just_completed=self.name))

        def need_help() -> None:
            """If the user asks to do something you don't know how to do, call this."""
            print(f"{self.name} needs help")
            self.context.data["redirecting"] = True
            parent.send_event(InterfaceAgentEvent(request=self.current_event.request, need_help=True))

        self.tools = [
            FunctionTool.from_defaults(fn=done),
            FunctionTool.from_defaults(fn=need_help)
        ]
        for t in tools:
            self.tools.append(t)

        agent_worker = FunctionCallingAgentWorker.from_tools(
            self.tools,
            llm=self.context.data["llm"],
            allow_parallel_tool_calls=False,
            system_prompt=self.system_prompt
        )
        self.agent = agent_worker.as_agent()

    def handle_event(self, ev: Event):
        self.current_event = ev

        response = str(self.agent.chat(ev.request))
        print(Fore.MAGENTA + str(response) + Style.RESET_ALL)

        # If they're sending us elsewhere we're done here
        if self.context.data["redirecting"]:
            self.context.data["redirecting"] = False
            return None

        # Otherwise, get some user input and then loop
        user_msg_str = input("> ").strip()
        return self.trigger_event(request=user_msg_str)

### --- Define Multi-Agent Workflow --- ###

class MultiAgentWorkflow(Workflow):
    @step(pass_context=True)
    async def initialize(self, ctx: Context, ev: InitializeEvent) -> InterfaceAgentEvent:
        ctx.data["initialized"] = None
        ctx.data["success"] = None
        ctx.data["redirecting"] = None
        ctx.data["overall_request"] = None

        # Initialize the LLM
        ctx.data["llm"] = OpenAI(model="gpt-4", temperature=0.4, api_key="<api>")

        return InterfaceAgentEvent()

    @step(pass_context=True)
    async def interface_agent(self, ctx: Context, ev: InterfaceAgentEvent | StartEvent) -> InitializeEvent | StopEvent | OrchestratorEvent:
        if ("initialized" not in ctx.data):
            return InitializeEvent()

        if ("InterfaceAgent" not in ctx.data):
            system_prompt = (
                f"You are a helpful assistant that assists users in querying video data using various tools and returns the timestamp present in the metadata of the embedding. "
                f"You can help the user with the following:\n"
                f"- Answer questions about image captions in video frames.\n"
                f"- Provide information from video transcripts.\n"
                f"- Describe objects detected in video frames using object detection.\n"
                f"Begin by asking the user how you can assist them."
            )

            agent_worker = FunctionCallingAgentWorker.from_tools(
                tools=[],
                llm=ctx.data["llm"],
                allow_parallel_tool_calls=False,
                system_prompt=system_prompt
            )
            ctx.data["InterfaceAgent"] = agent_worker.as_agent()

        interface_agent = ctx.data["InterfaceAgent"]
        if ctx.data["overall_request"] is not None:
            print("There's an overall request in progress.")
            last_request = ctx.data["overall_request"]
            ctx.data["overall_request"] = None
            return OrchestratorEvent(request=last_request)
        elif (ev.just_completed is not None):
            response = interface_agent.chat(f"The user has just completed the task: {ev.just_completed}")
        elif (ev.need_help):
            print("The previous process needs help with ", ev.request)
            return OrchestratorEvent(request=ev.request)
        else:
            response = interface_agent.chat("Hello!")

        print(Fore.MAGENTA + str(response) + Style.RESET_ALL)
        user_msg_str = input("> ").strip()
        return OrchestratorEvent(request=user_msg_str)

    @step(pass_context=True)
    async def orchestrator(self, ctx: Context, ev: OrchestratorEvent) -> InterfaceAgentEvent | ImageCaptioningEvent | TranscriptsEvent | YoloEvent | StopEvent:

        print(f"Orchestrator received request: {ev.request}")

        def emit_image_captioning() -> bool:
            """Call this if the user wants information from image captions."""
            print("__emitted: ImageCaptioningEvent")
            self.send_event(ImageCaptioningEvent(request=ev.request))
            return True

        def emit_transcripts() -> bool:
            """Call this if the user wants information from transcripts."""
            print("__emitted: TranscriptsEvent")
            self.send_event(TranscriptsEvent(request=ev.request))
            return True

        def emit_yolo() -> bool:
            """Call this if the user wants information from object detection."""
            print("__emitted: YoloEvent")
            self.send_event(YoloEvent(request=ev.request))
            return True

        def emit_interface_agent() -> bool:
            """Call this if you can't figure out what the user wants to do."""
            print("__emitted: interface")
            self.send_event(InterfaceAgentEvent(request=ev.request))
            return True

        def emit_stop() -> bool:
            """Call this if the user wants to stop or exit the system."""
            print("__emitted: stop")
            self.send_event(StopEvent())
            return True

        tools = [
            FunctionTool.from_defaults(fn=emit_image_captioning),
            # FunctionTool.from_defaults(fn=emit_transcripts),
            # FunctionTool.from_defaults(fn=emit_yolo),
            FunctionTool.from_defaults(fn=emit_interface_agent),
            FunctionTool.from_defaults(fn=emit_stop)
        ]

        system_prompt = (
            f"You are an orchestration agent.\n"
            f"Your job is to decide which tool or agent to run based on the user's request.\n"
            f"You have access to the following tools:\n"
            f"- image_captioning_tool: Use this for questions about image captions in video frames, timestamps.\n"
            # f"- transcripts_tool: Use this for questions about video transcripts or any audio requests.\n"
            # f"- yolo_tool: Use this for questions about objects detected in video frames.\n"
            f"- emit_interface_agent: Use this if you're unsure which tool to use.\n"
            f"- emit_stop: Use this if the user wants to stop or exit.\n"
            f"Decide which tool to call based on the user's request, and call it using its exact name.\n"
            f"If you did not call any tools, return the string 'FAILED' without quotes and nothing else."
        )

        agent_worker = FunctionCallingAgentWorker.from_tools(
            tools=tools,
            llm=ctx.data["llm"],
            allow_parallel_tool_calls=False,
            system_prompt=system_prompt
        )
        ctx.data["orchestrator"] = agent_worker.as_agent()

        orchestrator = ctx.data["orchestrator"]
        response = str(orchestrator.chat(ev.request))

        if response == "FAILED":
            print("Orchestration agent failed to return a valid tool; try again")
            return OrchestratorEvent(request=ev.request)

    @step(pass_context=True)
    async def image_captioning(self, ctx: Context, ev: ImageCaptioningEvent) -> InterfaceAgentEvent:

        print(f"Image Captioning received request: {ev.request}")

        if ("image_captioning_agent" not in ctx.data):
            system_prompt = (
                f"You are a helpful assistant that provides information based on image captions from video frames give timestamps.\n"
                f"Use the 'image_captioning_tool' to retrieve information.\n"
                f"Once you have retrieved the information, you *must* call the tool named 'done' to signal that you are done.\n"
                f"If the user asks for something outside your capabilities, call the tool 'need_help'."
            )

            ctx.data["image_captioning_agent"] = BaseAgent(
                name="Image Captioning Agent",
                parent=self,
                tools=[image_captioning_tool],
                context=ctx,
                system_prompt=system_prompt,
                trigger_event=ImageCaptioningEvent
            )

        return ctx.data["image_captioning_agent"].handle_event(ev)

    @step(pass_context=True)
    async def transcripts(self, ctx: Context, ev: TranscriptsEvent) -> InterfaceAgentEvent:

        print(f"Transcripts received request: {ev.request}")

        if ("transcripts_agent" not in ctx.data):
            system_prompt = (
                f"You are a helpful assistant that provides information based on video transcripts.\n"
                f"Use the 'transcripts_tool' to retrieve information.\n"
                f"Once you have retrieved the information, you *must* call the tool named 'done' to signal that you are done.\n"
                f"If the user asks for something outside your capabilities, call the tool 'need_help'."
            )

            ctx.data["transcripts_agent"] = BaseAgent(
                name="Transcripts Agent",
                parent=self,
                tools=[transcripts_tool],
                context=ctx,
                system_prompt=system_prompt,
                trigger_event=TranscriptsEvent
            )

        return ctx.data["transcripts_agent"].handle_event(ev)

    @step(pass_context=True)
    async def yolo(self, ctx: Context, ev: YoloEvent) -> InterfaceAgentEvent:

        print(f"YOLO received request: {ev.request}")

        if ("yolo_agent" not in ctx.data):
            system_prompt = (
                f"You are a helpful assistant that provides information based on objects detected in video frames using object detection.\n"
                f"Use the 'yolo_tool' to retrieve information.\n"
                f"Once you have retrieved the information, you *must* call the tool named 'done' to signal that you are done.\n"
                f"If the user asks for something outside your capabilities, call the tool 'need_help'."
            )

            ctx.data["yolo_agent"] = BaseAgent(
                name="YOLO Agent",
                parent=self,
                tools=[yolo_tool],
                context=ctx,
                system_prompt=system_prompt,
                trigger_event=YoloEvent
            )

        return ctx.data["yolo_agent"].handle_event(ev)

draw_all_possible_flows(MultiAgentWorkflow, filename="multi-agent-workflow.html")

### --- Run the Workflow --- ###

async def main():
    c = MultiAgentWorkflow(timeout=1200, verbose=True)
    result = await c.run()
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
