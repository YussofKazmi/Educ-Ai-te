from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import SerperDevTool

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Educaite():
    """Educaite crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        
        return Agent(
            config=self.agents_config['research_agent'], # type: ignore[index]
            verbose=True
        )

    @agent
    def knowledge_compiler(self) -> Agent:
        return Agent(
            config=self.agents_config['knowledge_agent'], # type: ignore[index]
            verbose=True
        )



    @agent
    def educator(self) -> Agent:
        return Agent(
            config=self.agents_config['educator_agent'], # type: ignore[index]
            verbose=True
        )
        
        
    @agent
    def quiz_creator(self) -> Agent:
        return Agent(
            config=self.agents_config['quiz_agent'], # type: ignore[index]
            verbose=True
        )
        
    @agent
    def result_delivery(self) -> Agent:
        return Agent(
            config=self.agents_config['result_delivery_agent'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['gather_information_task'], # type: ignore[index]
        )

    @task
    def compile_knowledge(self) -> Task:
        return Task(
            config=self.tasks_config['compile_knowledge_task'], # type: ignore[index]
            output_file='report.md'
        )

    @task
    def create_guide(self) -> Task:
        return Task(
            config=self.tasks_config['create_guide_task'], # type: ignore[index]
            #output_file='report.md'
        )

    @task
    def generate_quiz(self) -> Task:
        return Task(
            config=self.tasks_config['generate_quiz_task'], # type: ignore[index]
            #output_file='report.md'
        )

    @task
    def assemble_result(self) -> Task:
        return Task(
            config=self.tasks_config['assemble_result_task'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Educaite crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
