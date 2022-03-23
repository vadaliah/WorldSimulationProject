from copy import deepcopy
from queue import PriorityQueue

import pandas as pd

resourceweights = {}


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
        self.children = []

    def __str__(self):
        # resource_str= '-'.join(self.resources)
        return f"children_cnt: {len(self.children)}"

    def haschildren(self):
        return self.children

    def generate_successor(self, frontier, schedule_queue, depth, actions):
        #      world_state = deepcopy(node.world)
        parent = self
        parent.children = []
        for action in actions:
            world_successor = []
            country_successor = {}
            for country in self.world:
                if (verify_required_resources(country, action)):
                    print(
                        f"Applying Action:{action['Action']} Template:{action['TemplateName']} for county:{country['Country']}")
                    country_successor = apply_template(country, action)
                else:
                    print(
                        f"county:{country['Country']} does not have sufficient resources for applying Template:{action['TemplateName']}")
                world_successor.append(country_successor)
                child = Node(parent, world_successor)
                child.calculate_state_quality(resourceweights)
                schedule_str = ':'.join(
                    ['Depth', str(depth), 'State_quality', str(round(child.state_quality)), 'Template',
                     action['TemplateName']])
                values = list(map(lambda key: country[key], country.keys()))
                parent_str = '--'.join('{} : {}'.format(key, value) for key, value in country.items())
                child_str = '--'.join('{} : {}'.format(key, value) for key, value in country_successor.items())
                schedule_str += ':'.join([' Parent', parent_str, ' Child', child_str])
                #    schedule_str+= ':'.join()
                schedule_queue.put(schedule_str, child)
                # schedule_queue.put(schedule_str, child)
                frontier.put((-1 * child.state_quality, child))
                parent.children.append(child)

    # Calculate State Quality of the Node
    def calculate_state_quality(self, resourceweights):
        state_quality = 0
        for country in self.world:
            for key, value in country.items():
                if resourceweights.get(key) is not None:
                    state_quality += country[key] * resourceweights[key]
                else:
                    continue
        print(f'Node state_quality:{state_quality}')
        self.state_quality = state_quality


def apply_template(country, action):
    # Update Country resources with Subtrtacting Template input resources
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


def print_schedule(schedulequeue, output_schedule_filename):
    with open(output_schedule_filename, "w") as external_file:
        print("Printing Schedule", file=external_file)
        while not schedulequeue.empty():
            next_item = schedulequeue.get()
            print(next_item, file=external_file)
    external_file.close()


# Load World - Country Resources from Input file
def load_initial_state(country_resource_filename):
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


def load_resouce_weights(resource_filename):
    df = pd.read_csv(resource_filename)
    # print (df.columns.tolist())

    for i in range(len(df)):
        resourceweights[df.loc[i][0]] = df.loc[i][1]
    return resourceweights


def load_templates(template_filename):
    actions = []
    df = pd.read_csv(template_filename)
    # print (df.columns.tolist())
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


def my_country_scheduler(my_country_name, resource_filename, country_resources_filename, template_filename,
                         output_schedule_filename, num_output_schedules, depth_bound, frontier_max_size):
    world = load_initial_state(country_resources_filename)
    resourceweights = load_resouce_weights(resource_filename)
    print(resourceweights)
    actions = load_templates(template_filename)
    root_node = Node(None, world)
    root_node.calculate_state_quality(resourceweights)
    frontier = PriorityQueue()
    schedule_queue = PriorityQueue()
    current_depth = 0
    schedule_str = ':'.join(['Depth', str(current_depth), 'State_quality', str(root_node.state_quality), 'ROOT Node'])
    schedule_queue.put(schedule_str, root_node)
    frontier.put((-1 * root_node.state_quality, root_node))
    while not frontier.empty() and current_depth <= depth_bound:
        node = frontier.get()[1]
        current_depth += 1
        node.generate_successor(frontier, schedule_queue, current_depth, actions)

    print_schedule(schedule_queue, output_schedule_filename)


def main():
    my_country_scheduler('Atlantis', r'resourcedefination.csv',
                         r'initialcountry.csv',
                         r'TransformTemplates.csv',
                         r'Results.txt', 10, 2, 500)


if __name__ == "__main__":
    main()
