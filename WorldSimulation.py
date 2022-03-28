import math
from copy import deepcopy
from queue import PriorityQueue

import pandas as pd

resourceweights = {}
# Calculation constants
Gamma = 0.95
C = -10
K = 1
X_0 = 0


class Action:
    def __init__(self, templatename, action):
        self.TemplateName = templatename
        self.Operation = action
        self.input_resources = []
        self.output_resources = []

    def add_input_resource(self, resource):
        self.input_resources.append(resource)

    def add_output_resource(self, resource):
        self.output_resources.append(resource)

    def __str__(self):
        # resource_str= '-'.join(self.resources)
        return f"Action: {self.TemplateName}"


class Node:
    def __init__(self, parent, world):
        self.parent = parent
        self.world = world
        self.state_quality = 0
        self.score = 0
        self.children = []

    def __str__(self):
        # resource_str= '-'.join(self.resources)
        return f"children_cnt: {len(self.children)}"

    def get_first_child(self):
        """
        :return: Return First Child with max state_quality
        """
        node = self
        if node.children:
            return (node.children.sort(key=lambda x: x.state_quality, reverse=True))[0]
        else:
            return None

    def calculate_schedule_probability(self):
        """
        :return: Schedule Probability sco:L / 1 + math.e ** (-K * (x - X_0))
        """
        node = self
        L = 1
        x = node.calculate_discounted_reward()
        score = L / 1 + math.e ** (-K * (x - X_0))
        self.score = score
        return score

    def calculate_expected_utility(self):
        """
         # calculate_expected_utility function implements the following:
        # EU(c_i, s_j) = (P(s_j) * DR(c_i, s_j)) + ((1-P(s_j)) * C), where c_i = self
        :return:
        """
        node = self
        P = node.calculate_schedule_probability()
        DR = node.calculate_discounted_reward()
        return (P * DR) + ((1 - P) * C)

    def get_children_count(self):
        node = self
        if node.children is None:
            return 0
        return 1 + len(node.children)

    def get_state_quality(self):
        """
        state_quality Getter
        :return: node.state_quality
        """
        node = self
        if node is None:
            return 0
        return node.state_quality

    def get_probability_score(self):
        """
        Probablity_score Getter
        :return: node.score
        """
        node = self
        if node is None:
            return 0
        return node.score

    def calculate_undiscounted_reward(self):
        """
        # From the requirements, this function implements the following equation:
        # R(c_i, s_j) = Q_end(c_i, s_j) – Q_start(c_i, s_j) to a country c_i of a schedule s_j.
        :param :
        :return: state_quality(node) - state_quality(start)
        """
        node = self
        sq1 = (node.get_first_child()).get_state_quality() if (node.children) else 0

        return node.get_state_quality() - sq1

    def calculate_discounted_reward(self):
        """
        Calculate discounted_reward using following equation:
        # DR(c_i, s_j) = gamma^N * (Q_end(c_i, s_j) – Q_start(c_i, s_j)), where 0 <= gamma < 1.
        :return: (Gamma ** count) * node.calculate_undiscounted_reward()
        """
        node = self
        count = node.get_children_count()
        return (Gamma ** count) * node.calculate_undiscounted_reward()

    def generate_successor(self, frontier, schedule_queue, depth, actions):
        """
            Function that generates applies Action on Each Countries within Node.World to generate Successor Nodes
        """
        parent = self
        parent.children = []
        for action in actions:
            world_successor = []
            country_successor = {}
            for country in self.world:
                if verify_required_resources(country,
                                             action):  # Verify Country has adaquate resources to apply Action template
                    print(
                        f"Applying Template:{action['TemplateName']} for county:{country['Country']}")
                    country_successor = apply_template(country, action)
                else:
                    print(
                        f"county:{country['Country']} does not have sufficient resources for:{action['TemplateName']}")
                world_successor.append(country_successor)
                child = Node(parent, world_successor)
                child.calculate_state_quality(resourceweights)
                child.calculate_schedule_probability()
                schedule_str = ':'.join(
                    ['Depth', str(depth), 'State_Q:uality', str(round(child.get_state_quality())), 'Template',
                     action['TemplateName']])
                #                  ['Depth', str(depth), 'State_quality', str(round(child.state_quality)), 'Template', action['TemplateName']])
                parent_str = '--'.join('{} : {}'.format(key, value) for key, value in country.items())
                child_str = '--'.join('{} : {}'.format(key, value) for key, value in country_successor.items())
                schedule_str += ':'.join([' Parent', parent_str, ' Child', child_str])
                #    schedule_str+= ':'.join()
                schedule_queue.put(schedule_str, child)
                # schedule_queue.put(schedule_str, child)
                frontier.put((-1 * child.state_quality, child))
                parent.children.append(child)

    def calculate_state_quality(self, resourceweights):
        """
            # Function to calculate Node State Quality
            :param
            Input: Node object
            Input: resourceweights: Map object containing Resource Weight definition
            # Part1 State quality function uses static Weights associated per resource type
            # For Part , intend to enhance weights using country specific methodology
        """
        state_quality = 0
        for country in self.world:
            for resource, amount in country.items():
                if resourceweights.get(resource) is not None:
                    state_quality += country[resource] * resourceweights[resource]
                else:
                    continue
        self.state_quality = state_quality


def apply_template(country, action):
    """
    Function to apply action template on a country object
    Country Resources are reduced as per Template Input Resource definition
    Country Resources are added as per Template Out Resource defination
    Additional Country Resources are created for non existing Template Out resources
    :param country:
    :param action:
    :return:
    """
    country_successor = deepcopy(country)
    for key, value in action.items():
        if key.startswith('IN_') and action[key] > 0:
            country_key = key.removeprefix('IN_')
            if (country_successor.get(country_key) is not None) and (action[key] <= country_successor[country_key]):
                country_successor[country_key] -= action[key]
        elif key.startswith('OUT_') and action[key] > 0:
            country_key = key.removeprefix('OUT_')
            country_successor[country_key] += action[key] if country_successor.get(country_key) is not None else \
                action[key]
        else:
            continue
    return country_successor


# from the requirements, uses the logistic function:
# https://en.wikipedia.org/wiki/Logistic_function
def calculate_schedule_probability(node):
    L = 1
    K = 1
    x = node.calculate_discounted_reward()

    return L / 1 + math.e ** (-K * (x - X_0))


def print_schedule(schedulequeue, output_schedule_filename):
    """
    Function to print Schedules to an output file
    :param schedulequeue:
    :param output_schedule_filename:
    :return:
    """
    print(f"Printing Schedules to Output file: {output_schedule_filename} ")
    with open(output_schedule_filename, "w") as external_file:
        print("Printing Schedule", file=external_file)
        while not schedulequeue.empty():
            next_item = schedulequeue.get()
            print(next_item, file=external_file)
    external_file.close()


# Load World - Country Resources from Input file
def load_initial_state(country_resource_filename):
    """
    Function to load initial World Resource state from an input CSV file
    Panda dataframe is used to read CSV file for creating Country objects
    :param country_resource_filename:
    :return:
    """
    world = []
    df = pd.read_csv(country_resource_filename)
    for i in range(len(df)):
        country = {}
        for j in range(len(df.columns)):
            country[df.columns[j]] = df.loc[i][j]
        print(country)
        world.append(country)
    # df = pd.read_csv(initial_state_file_name)
    # country[df]
    return world


def load_resource_weights(resource_filename):
    """
    Function to populate resourceweights map of Resource Weights from an input CSV file
    :param resource_filename:
    :return:
    """
    df = pd.read_csv(resource_filename)
    # print (df.columns.tolist())

    for i in range(len(df)):
        resourceweights[df.loc[i][0]] = df.loc[i][1]
    return resourceweights


def load_templates(template_filename):
    """
     Function to populate actions list  of Action Templates from an input CSV file
     Actions object consists of Action[Transform/Transfer], TemplateName, Input Resources : Resources required as Inputs
     and Output Resources: Resources generated as a Byproduct
     Panda DataFrame is used to Read CSV file and populate Action dictionary object
    :param template_filename:
    :return:
    """
    actions = []
    df = pd.read_csv(template_filename)
    for i in range(len(df)):
        template = {}
        for j in range(len(df.columns)):
            template[df.columns[j]] = df.loc[i][j]
        actions.append(template)
    return actions


def verify_required_resources(country, action):
    for key, value in action.items():
        if key.startswith('IN_') and action[key] > 0:
            country_key = key.removeprefix('IN_')
            if (country.get(country_key) is None) or (action[key] > country[country_key]):
                print(f"Country: {country['Country']} Resource :{country_key} failed resource check")
                return False
        else:
            continue
    return True


def my_country_scheduler(my_country_name, resource_weight_filename, country_resources_filename,
                         action_template_filename,
                         output_schedule_filename, num_output_schedules, depth_bound, frontier_max_size):
    """
    u
    :param my_country_name:  My Country
    :param resource_weight_filename: Input CSV file for Resource Weights
    :param country_resources_filename:  Input CSV File for World Countries Resource definition
    :param action_template_filename: Input CVS file for Templates
    :param output_schedule_filename: Output CSV file for generated Schedules
    :param num_output_schedules:  Threshold for Output_schedules
    :param depth_bound:  Threshold for Schedule Depth
    :param frontier_max_size: Threshold for Frontier size
    :return:
    """
    world = load_initial_state(country_resources_filename)
    resourceweights = load_resource_weights(resource_weight_filename)
    actions = load_templates(action_template_filename)
    frontier = PriorityQueue()
    schedule_queue = PriorityQueue()
    current_depth = 0
    root_node = Node(None, world)
    root_node.calculate_state_quality(resourceweights)
    score = root_node.calculate_schedule_probability()
    frontier.put((-1 * score, root_node))

    schedule_str = ':'.join(
        ['Depth', str(current_depth), 'Probability_score', str(root_node.get_probability_score()), 'ROOT Node'])
    schedule_queue.put(schedule_str, root_node)

    while not frontier.empty() and current_depth <= depth_bound:
        node = frontier.get()[1]
        current_depth += 1
        node.generate_successor(frontier, schedule_queue, current_depth, actions)

    print_schedule(schedule_queue, output_schedule_filename)


def main():
    my_country_scheduler('Atlantis', r'resourcedefination.csv',
                         r'initialcountry.csv',
                         r'TransformTemplates.csv',
                         r'Schedule_results.txt', 10, 2, 500)


if __name__ == "__main__":
    main()
