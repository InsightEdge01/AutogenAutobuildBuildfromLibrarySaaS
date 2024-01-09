import autogen
import json
from autogen.agentchat.contrib.agent_builder import AgentBuilder

# 1.specify a config_path, default_llm_config that include backbone LLM configurations.
config_path = 'OAI_CONFIG_LIST'
config_list = autogen.config_list_from_json(config_path)
llm_config = {'temperature': 0}

#function to start task
def start_task(execution_task: str, agent_list: list):
    group_chat = autogen.GroupChat(agents=agent_list, messages=[], max_round=12)
    manager = autogen.GroupChatManager(groupchat=group_chat, llm_config={"config_list": config_list, **llm_config})
    agent_list[0].initiate_chat(manager, message=execution_task)


#define a prompt template and a list of agents' name:
AGENT_SYS_MSG_PROMPT = """Considering the following position:

POSITION: {position}

What requirements should this position be satisfied?

Hint:
# Your answer should be in one sentence.
# Your answer should be natural, starting from "As a ...".
# People with the above position need to complete a task given by a leader or colleague.
# People will work in a group chat, solving tasks with other people with different jobs.
# The modified requirement should not contain the code interpreter skill.
# Coding skill is limited to Python.
"""

#generate an agent library
position_list = [
    "Marketing ",
    "Sales",
    "Software_Developer",
    "Data_Analyst",
    "Product Manager",
    "Critic Manager",
    "Programmer",
    "IT_Specialist",
    "Cybersecurity_Expert",
    "Artificial_Intelligence_Engineer",
    "Financial_Analyst",
]

#prompt the model to generate each agent's profile
build_manager = autogen.OpenAIWrapper(config_list=config_list)
sys_msg_list = []

for pos in position_list:
    resp_agent_sys_msg = (
        build_manager.create(
            messages=[
                {
                    "role": "user",
                    "content": AGENT_SYS_MSG_PROMPT.format(
                        position=pos,
                        default_sys_msg=autogen.AssistantAgent.DEFAULT_SYSTEM_MESSAGE,
                    ),
                }
            ]
        )
        .choices[0]
        .message.content
    )
    sys_msg_list.append({"name": pos, "profile": resp_agent_sys_msg})

    print(sys_msg_list)

#save the generated agents' information into a json file.
json.dump(sys_msg_list, open("./agent_library_example.json", "w"), indent=4)

library_path_or_json = "./agent_library_example.json"
building_task = """Conceptualize a Software as a Service (SaaS) solution tailored to optimize and streamline supply chain management processes for businesses. Consider features that enhance real-time visibility, collaboration,
and efficiency within the supply chain, emphasizing cloud-based accessibility and scalability."""


#we can call the build_from_library from the AgentBuilder to generate a list of agents 
#from the library and let them complete the user's execution_task in a group chat.
new_builder = AgentBuilder(
    config_file_or_env=config_path, builder_model="gpt-4-1106-preview", agent_model="gpt-4-1106-preview"
)
agent_list, _ = new_builder.build_from_library(building_task, library_path_or_json,llm_config)
start_task(
    execution_task="""Conceptualize a Software as a Service (SaaS) solution tailored to optimize and streamline supply chain management processes for businesses. Consider features that enhance real-time visibility, collaboration, and efficiency within the supply chain, emphasizing 
    cloud-based accessibility and scalability""",
    agent_list=agent_list,
)
new_builder.clear_all_agents()

