from autogen_agentchat.agents import AssistantAgent, UserProxyAgent, MessageFilterAgent, MessageFilterConfig, PerSourceFilter
from autogen_agentchat.teams import GraphFlow, DiGraphBuilder
from autogen_agentchat.conditions import MaxMessageTermination

def make_generator_agents(model_client):
    # Agent 1: Static instruction mix analysis of kernel
    static_instr_mix_agent = AssistantAgent(
        name="StaticInstructionMixAgent",
        system_message=(
            "You are an expert in static code analysis for CUDA C/C++ programs. "
            "Given source code as a string, the target CUDA kernel name, the executable input arguments, "
            "and the kernel's grid/block size and arguments, count the number of each operation type for a "
            "single kernel invocation. For each operation, report in the following format:\n"
            "SP FLOP ADD: XXX\nSP FLOP MUL: XXX\nSP FLOP DIV: XXX\nSP FLOP FMA: XXX\n"
            "DP FLOP ADD: XXX\nDP FLOP MUL: XXX\nDP FLOP DIV: XXX\nDP FLOP FMA: XXX\n"
            "INT ADD: XXX\nINT MUL: XXX\nINT DIV: XXX\nINT FMA: XXX\n"
            "Fused multiply-adds (FMA) should be counted as 1 here; do not double-count them. "
            "Analyze properly using loop bounds, kernel launch size, and executable argument values."
        ),
        model_client=model_client,
    )

    # Agent 2: Data type and allocation analysis
    data_allocation_agent = AssistantAgent(
        name="KernelDataAllocationAgent",
        system_message=(
            "You analyze CUDA C/C++ source code and estimate the amount of global memory allocated "
            "for integer, single-precision float, and double-precision float data for a single invocation "
            "of the target CUDA kernel. Use the kernel arguments and executable input arguments to estimate "
            "the bytes allocated. Output in the format:\n"
            "INT data allocation: XXX bytes\nSP Float data allocation: XXX bytes\nDP Float data allocation: XXX bytes"
        ),
        model_client=model_client,
    )

    # Agent 3: Global memory reads
    memory_read_agent = AssistantAgent(
        name="GlobalMemoryReadAgent",
        system_message=(
            "You identify and count the number of strictly data READS from global memory performed by a single invocation "
            "of the target CUDA kernel. For each type, output:\n"
            "INTOP Reads: XXXX\nSP FLOP Reads: XXXX\nDP FLOP Reads: XXXX"
        ),
        model_client=model_client,
    )

    # Agent 4: Global memory writes
    memory_write_agent = AssistantAgent(
        name="GlobalMemoryWriteAgent",
        system_message=(
            "You identify and count the number of strictly data WRITES to global memory performed by a single invocation "
            "of the target CUDA kernel. For each type, output:\n"
            "INTOP Writes: XXXX\nSP FLOP Writes: XXXX\nDP FLOP Writes: XXXX"
        ),
        model_client=model_client,
    )

    # Agent 5: Data reuse characterization
    data_reuse_agent = AssistantAgent(
        name="KernelDataReuseAgent",
        system_message=(
            "You analyze data use and re-use patterns in the target CUDA kernel, classifying each operation type as "
            "'DENSE', 'MIXED', 'SPARSE', or 'RANDOM' in terms of data access/reuse."
        ),
        model_client=model_client,
    )

    return (
        static_instr_mix_agent,
        data_allocation_agent,
        memory_read_agent,
        memory_write_agent,
        data_reuse_agent,
    )

def make_review_agents(model_client):
    # Reviewers for each generator agent
    static_instr_mix_reviewer_agent = AssistantAgent(
        name="StaticInstructionMixReviewerAgent",
        system_message=(
            "You are a reviewer. Assess the static instruction mix analysis for accuracy, completeness, "
            "and plausible loop bounds. Respond with 'APPROVE' if correct, or 'REVISE' with feedback if not."
        ),
        model_client=model_client,
    )
    data_allocation_reviewer_agent = AssistantAgent(
        name="DataAllocationReviewerAgent",
        system_message=(
            "You are a reviewer. Assess the data allocation output for type correctness and plausibility. "
            "Respond with 'APPROVE' if correct, or 'REVISE' with feedback if not."
        ),
        model_client=model_client,
    )
    memory_read_reviewer_agent = AssistantAgent(
        name="MemoryReadReviewerAgent",
        system_message=(
            "You are a reviewer. Assess the memory read counts for accuracy and completeness. "
            "Respond with 'APPROVE' if correct, or 'REVISE' with feedback if not."
        ),
        model_client=model_client,
    )
    memory_write_reviewer_agent = AssistantAgent(
        name="MemoryWriteReviewerAgent",
        system_message=(
            "You are a reviewer. Assess the memory write counts for accuracy and completeness. "
            "Respond with 'APPROVE' if correct, or 'REVISE' with feedback if not."
        ),
        model_client=model_client,
    )
    data_reuse_reviewer_agent = AssistantAgent(
        name="DataReuseReviewerAgent",
        system_message=(
            "You are a reviewer. Assess the data reuse categorization for accuracy. "
            "Respond with 'APPROVE' if correct, or 'REVISE' with feedback if not."
        ),
        model_client=model_client,
    )
    return (
        static_instr_mix_reviewer_agent,
        data_allocation_reviewer_agent,
        memory_read_reviewer_agent,
        memory_write_reviewer_agent,
        data_reuse_reviewer_agent,
    )

def make_message_filter_agent(name, source_generator, source_reviewer):
    # Filter config: accept the first message from generator and last from reviewer
    return MessageFilterAgent(
        name=name,
        wrapped_agent=source_generator,
        filter=MessageFilterConfig(
            per_source=[
                PerSourceFilter(source="user", position="first", count=1),
                PerSourceFilter(source=source_generator.name, position="last", count=3),
                PerSourceFilter(source=source_reviewer, position="last", count=3),
            ]
        ),
    )


def build_graphflow(model_client):
    start_agent = AssistantAgent(
        "DummyInitialRequestAgent",
        model_client=model_client,
        system_message=f"""You are a dummy agent. You don't say or return any text. Reply ONLY with the empty string.""",
    )

    (
        static_instr_mix_agent,
        data_allocation_agent,
        memory_read_agent,
        memory_write_agent,
        data_reuse_agent,
    ) = make_generator_agents(model_client)

    (
        static_instr_mix_reviewer_agent,
        data_allocation_reviewer_agent,
        memory_read_reviewer_agent,
        memory_write_reviewer_agent,
        data_reuse_reviewer_agent,
    ) = make_review_agents(model_client)

    # MessageFilterAgents for each review chain, sources are generator and reviewer agent names
    static_instr_mix_filter_agent = make_message_filter_agent(
        "StaticInstructionMixFilterAgent",
        source_generator=static_instr_mix_agent,
        source_reviewer="StaticInstructionMixReviewerAgent"
    )
    data_allocation_filter_agent = make_message_filter_agent(
        "DataAllocationFilterAgent",
        source_generator=data_allocation_agent,
        source_reviewer="DataAllocationReviewerAgent"
    )
    memory_read_filter_agent = make_message_filter_agent(
        "MemoryReadFilterAgent",
        source_generator=memory_read_agent,
        source_reviewer="MemoryReadReviewerAgent"
    )
    memory_write_filter_agent = make_message_filter_agent(
        "MemoryWriteFilterAgent",
        source_generator=memory_write_agent,
        source_reviewer="MemoryWriteReviewerAgent"
    )
    data_reuse_filter_agent = make_message_filter_agent(
        "DataReuseFilterAgent",
        source_generator=data_reuse_agent,
        source_reviewer="DataReuseReviewerAgent"
    )

    summary_agent = AssistantAgent(
        name="KernelSummaryAgent",
        system_message=(
            "You synthesize the reports from the other agents to predict:\n"
            "- integer operations\n- single-precision floating point operations\n"
            "- double-precision floating point operations\n- number of bytes read\n- number of bytes written\n"
            "Count FMA as 2 operations. Sum all appropriate operation types. For bytes, "
            "combine read/write counts appropriately. Output must be:\n"
            "integer operations: XXXX\nsingle-precision floating point operations: XXXX\n"
            "double-precision floating point operations: XXXX\nnumber of bytes read: XXXX\n"
            "number of bytes written: XXXX"
        ),
        model_client=model_client,
    )

    # Build the agent graph as described
    builder = DiGraphBuilder()

    #builder.add_node(user_input_agent)
    builder.add_node(start_agent)
    builder.add_node(static_instr_mix_agent)
    builder.add_node(static_instr_mix_reviewer_agent)
    builder.add_node(static_instr_mix_filter_agent)

    builder.add_node(data_allocation_agent)
    builder.add_node(data_allocation_reviewer_agent)
    builder.add_node(data_allocation_filter_agent)

    builder.add_node(memory_read_agent)
    builder.add_node(memory_read_reviewer_agent)
    builder.add_node(memory_read_filter_agent)

    builder.add_node(memory_write_agent)
    builder.add_node(memory_write_reviewer_agent)
    builder.add_node(memory_write_filter_agent)

    builder.add_node(data_reuse_agent)
    builder.add_node(data_reuse_reviewer_agent)
    builder.add_node(data_reuse_filter_agent)

    builder.add_node(summary_agent)

    #builder.set_entry_point(user_input_agent)

    # User input fans out to all initial generator agents
    builder.add_edge(start_agent, static_instr_mix_agent)
    builder.add_edge(start_agent, data_allocation_agent)
    builder.add_edge(start_agent, memory_read_agent)
    builder.add_edge(start_agent, memory_write_agent)
    builder.add_edge(start_agent, data_reuse_agent)

    # Validation/revision/review loops for each generator agent
    builder.add_edge(static_instr_mix_agent, static_instr_mix_reviewer_agent)
    builder.add_edge(static_instr_mix_reviewer_agent, static_instr_mix_agent, condition="REVISE")
    builder.add_edge(static_instr_mix_reviewer_agent, static_instr_mix_filter_agent, condition="APPROVE")

    builder.add_edge(data_allocation_agent, data_allocation_reviewer_agent)
    builder.add_edge(data_allocation_reviewer_agent, data_allocation_agent, condition="REVISE")
    builder.add_edge(data_allocation_reviewer_agent, data_allocation_filter_agent, condition="APPROVE")

    builder.add_edge(memory_read_agent, memory_read_reviewer_agent)
    builder.add_edge(memory_read_reviewer_agent, memory_read_agent, condition="REVISE")
    builder.add_edge(memory_read_reviewer_agent, memory_read_filter_agent, condition="APPROVE")

    builder.add_edge(memory_write_agent, memory_write_reviewer_agent)
    builder.add_edge(memory_write_reviewer_agent, memory_write_agent, condition="REVISE")
    builder.add_edge(memory_write_reviewer_agent, memory_write_filter_agent, condition="APPROVE")

    builder.add_edge(data_reuse_agent, data_reuse_reviewer_agent)
    builder.add_edge(data_reuse_reviewer_agent, data_reuse_agent, condition="REVISE")
    builder.add_edge(data_reuse_reviewer_agent, data_reuse_filter_agent, condition="APPROVE")

    # All filtered/approved outputs go to the summary agent
    builder.add_edge(static_instr_mix_filter_agent, summary_agent)
    builder.add_edge(data_allocation_filter_agent, summary_agent)
    builder.add_edge(memory_read_filter_agent, summary_agent)
    builder.add_edge(memory_write_filter_agent, summary_agent)
    builder.add_edge(data_reuse_filter_agent, summary_agent)

    builder.set_entry_point(start_agent)

    graph = builder.build()

    workflow = GraphFlow(
        graph=graph,
        termination_condition=MaxMessageTermination(20),
        participants=builder.get_participants()
    )
    return workflow
