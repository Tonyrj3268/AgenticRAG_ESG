import asyncio
import time
import warnings
from types import MethodType
from typing import Any, Callable, List, Optional, Union

from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.core.workflow.decorators import StepConfig, step
from llama_index.core.workflow.errors import WorkflowDone
from llama_index.core.workflow.utils import get_steps_from_instance

dispatcher = get_dispatcher(__name__)


class ImprovedWorkflow(Workflow):
    def replicate_step(
        self, original_name: str, new_name: str, stepwise: bool = False
    ) -> None:
        all_steps = self._get_steps()

        if original_name not in all_steps:
            raise ValueError(f"Step {original_name} does not exist")
        if new_name in all_steps:
            raise ValueError(f"Step {new_name} already exists")
        step_func = all_steps[original_name]

        step_config = getattr(step_func, "__step_config", None)

        async def _task(
            name: str,
            queue: asyncio.Queue,
            step: Callable,
            config: StepConfig,
        ) -> None:
            while True:
                ev = await queue.get()
                if type(ev) not in config.accepted_events:
                    continue

                # do we need to wait for the step flag?
                if stepwise:
                    await self._step_flags[name].wait()

                    # clear all flags so that we only run one step
                    for flag in self._step_flags.values():
                        flag.clear()

                if self._verbose and name != "_done":
                    print(f"Running step {name}")

                # run step
                args = []
                if config.pass_context:
                    args.append(self.get_context(name))
                args.append(ev)

                # - check if its async or not
                # - if not async, run it in an executor
                instrumented_step = dispatcher.span(step)

                if asyncio.iscoroutinefunction(step):
                    new_ev = await instrumented_step(*args)
                else:
                    new_ev = await asyncio.get_event_loop().run_in_executor(
                        None, instrumented_step, *args
                    )

                if self._verbose and name != "_done":
                    if new_ev is not None:
                        print(f"Step {name} produced event {type(new_ev).__name__}")
                    else:
                        print(f"Step {name} produced no event")

                # handle the return value
                if new_ev is None:
                    continue

                # Store the accepted event for the drawing operations
                self._accepted_events.append((name, type(ev).__name__))

                if not isinstance(new_ev, Event):
                    warnings.warn(
                        f"Step function {name} returned {type(new_ev).__name__} instead of an Event instance."
                    )
                else:
                    self.send_event(new_ev)

        self._tasks.add(
            asyncio.create_task(
                _task(new_name, self._queues[original_name], step_func, step_config),
                name=new_name,
            )
        )


class QueryEvent(Event):
    query: str


class SubqueryEvent(Event):
    subquery: str


class SubqueryEvent_1(Event):
    subquery: str


class SubqueryEvent_2(Event):
    subquery: str


class ResponseEvent(Event):
    response: str


class CollectEvent(Event):
    pass


class SimplifiedQuerySplitWorkflow(Workflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ai_model = FakeAIModel()
        self.start_time = time.time()

    def log_time(self, message):
        print(f"{message} - {time.time() - self.start_time:.2f}s")

    @step()
    async def receive_query(self, ev: StartEvent) -> QueryEvent:
        self.log_time("Received query")
        main_query = ev.get("question", "Default question")
        return QueryEvent(query=main_query)

    @step()
    async def split_query(
        self, ev: QueryEvent
    ) -> CollectEvent | SubqueryEvent | StopEvent:
        self.log_time("Splitting query")

        # self.replicate_step("process_subquery", f"process_subquery_1")
        # self.replicate_step("process_subquery", f"process_subquery_2")

        for i in range(10):
            subquery = f"Subquery {i+1} from {ev.query}"
            self.send_event(SubqueryEvent(subquery=subquery))
        return CollectEvent()

    @step()
    async def process_subquery(self, ev: SubqueryEvent) -> ResponseEvent:
        response = await self.ai_model.process(ev.subquery)
        self.log_time(f"Processed {ev.subquery}")
        return ResponseEvent(response=response)

    @step()
    async def collect_responses(
        self, ev: ResponseEvent | CollectEvent
    ) -> StopEvent | None:
        if not hasattr(self, "responses"):
            self.responses = []
        if isinstance(ev, ResponseEvent):
            self.responses.append(ev.response)
            self.log_time(f"Collected response {len(self.responses)}/10")
        if len(self.responses) == 10:
            self.log_time("All responses collected")
            return StopEvent(result="Responses collected")
        return None


class FakeAIModel:
    async def process(self, query: str) -> str:
        await asyncio.sleep(0.5)  # Simulating processing time
        return f"Response to {query}"


async def run_workflow():
    workflow = SimplifiedQuerySplitWorkflow()
    result = await workflow.run(question="Test Question?")
    print(f"Final result: {result}")
    print(f"Total time: {time.time() - workflow.start_time:.2f}s")


asyncio.run(run_workflow())
