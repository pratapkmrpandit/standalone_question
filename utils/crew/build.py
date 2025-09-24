import os
import yaml
from dotenv import load_dotenv
from crewai import Agent, Task, Crew,LLM
from config.constant import Constant

def load_env():
    """Load environment variables from .env file"""
    load_dotenv()

def load_yaml(path):
    """Helper to load yaml files"""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_llm(model_name="gemini/gemini-2.0-flash"):
    """Return an LLM instance using GEMINI API key."""
    api_key = os.getenv("GEMINI_API_KEY")
    return LLM(api_key=api_key, model=model_name, temperature=0.1)

def create_crew():
    """Create Crew from agents.yaml and tasks.yaml"""
    # Load YAML
    agents_conf = load_yaml(Constant.CREW_FILE_PATH['agents'])['agents']
    tasks_conf = load_yaml(Constant.CREW_FILE_PATH['tasks'])['tasks']

    # Create Agents
    agents = {}
    llm= get_llm()
    for name, cfg in agents_conf.items():
        agents[name] = Agent(
            role=cfg['role'],
            goal=cfg['goal'],
            backstory=cfg['backstory'],
            allow_delegation=cfg.get('allow_delegation', False),
            verbose=cfg.get('verbose', False),
            llm=llm
        )

    # Create Tasks
    tasks = []
    for name, cfg in tasks_conf.items():
        # history_str = "\n".join([f"{item['role'].capitalize()}: {item['content']}" for item in inputs['history']])
        # description = cfg['description'].replace("{history}", history_str)
        tasks.append(Task(
            description=cfg['description'],
            expected_output=cfg['expected_output'],
            agent=agents[cfg['agent']],
        ))

    # Create Crew
    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        verbose=True
    )
    return crew

def execute_crew(inputs: dict):
    """Execute the crew with provided inputs"""
    load_env()
    crew = create_crew()
    result = crew.kickoff(inputs=inputs)
    return result
