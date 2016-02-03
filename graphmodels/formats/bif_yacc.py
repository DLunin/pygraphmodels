import ply.yacc as yacc
import numpy as np
from functools import wraps

# Get the token map from the lexer.  This is required.
from pygraphmodels.graphmodels.formats.bif_lex import tokens

def list_rule(rule_name, elem_name, empty_allowed=False):
    def rule(p):
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = [p[1]] + p[2]

    rule.__doc__ = """{0} : {1}
                          | {1} {0}""".format(rule_name, elem_name)
    if empty_allowed:
        rule.__doc__ += '\n|'
    rule.__name__ = 'p_' + rule_name
    return rule

def alternative_rule(rule_name, alternatives):
    alternatives = list(alternatives)
    assert len(alternatives) >= 0

    def rule(p):
        p[0] = p[1]

    rule.__doc__ = '{rule_name} : {alternative}'.format(rule_name=rule_name, alternative=alternatives[0])
    for alternative in alternatives[1:]:
        rule.__doc__ += '\n| {alternative}'.format(alternative=alternative)
    rule.__name__ = 'p_' + rule_name
    return rule

p_value = alternative_rule('value', ['WORD', 'FLOATING_POINT_LITERAL', 'DECIMAL_LITERAL'])
p_block = alternative_rule('block', ['variable_declaration', 'probability_declaration'])

p_list = list_rule('list', 'value', empty_allowed=False)
p_string = list_rule('string', 'WORD', empty_allowed=False)
p_property_list = list_rule('property_list', 'property_string', empty_allowed=False)
p_floating_point_list = list_rule('floating_point_list', 'FLOATING_POINT_LITERAL', empty_allowed=False)
p_blocks_list = list_rule('blocks_list', 'block', empty_allowed=False)

p_probability_content_item = alternative_rule('probability_content_item',
    ['property_string', 'probability_entry', 'probability_default_entry', 'probability_table'])

p_probability_content = list_rule('probability_content', 'probability_content_item', empty_allowed=False)

p_variable_content_item = alternative_rule('variable_content_item', ['property_string', 'variable_discrete'])
p_variable_content = list_rule('variable_content', 'variable_content_item', empty_allowed=False)


def p_property_string(p):
    "property_string : PROPERTY WORD string ';'"
    p[0] = 'property', {p[2]: ' '.join(p[3])}


def p_network_declaration(p):
    "network_declaration : NETWORK WORD network_content"
    p[0] = {'name': p[2], 'properties': p[3]}


def p_network_content(p):
    """network_content : '{' property_list '}'
                       | '{' '}'"""
    p[0] = {}
    if len(p) == 4:
        for t, prop in p[2]:
            p[0].update(prop)


def p_probability_variables_list(p):
    """probability_variables_list : '(' list ')'"""
    p[0] = p[2]


def p_probability_values_list(p):
    """probability_values_list : '(' list ')'"""
    p[0] = p[2]


def p_probability_entry(p):
    """probability_entry : probability_values_list floating_point_list ';'"""
    p[0] = 'probability_entry', (p[1], p[2])


def p_probability_default_entry(p):
    """probability_default_entry : DEFAULTVALUE floating_point_list ';'"""
    p[0] = 'default', p[2]


def p_probability_table(p):
    """probability_table : TABLEVALUES floating_point_list ';'"""
    p[0] = 'table', np.asarray(p[2], dtype='float')

def p_probability_declaration(p):
    """probability_declaration : PROBABILITY probability_variables_list '{' probability_content '}'"""
    p[0] = 'distribution', {'variables': p[2], 'properties': p[4]}


def p_variable_discrete(p):
    """variable_discrete : VARIABLETYPE DISCRETE '[' DECIMAL_LITERAL ']' '{' list '}' ';' """
    p[0] = 'discrete', {'n_values': p[4], 'values_list': p[7]}


def p_variable_declaration(p):
    """variable_declaration : VARIABLE WORD '{' variable_content '}' """
    p[0] = 'variable', (p[2], p[4])


def p_compilation_unit(p):
    """compilation_unit : network_declaration blocks_list"""
    p[0] = {}
    p[0].update(p[1])
    p[0].update({'variables': [], 'distributions': []})

    for block_type, block in p[2]:
        if block_type == 'variable':
            current_variable = {'name': block[0], 'properties': {}}
            for attr_type, attr in block[1]:
                if attr_type == 'property':
                    current_variable['properties'].update(attr)
                elif attr_type == 'discrete':
                    current_variable.update(attr)
            p[0]['variables'].append(current_variable)
        elif block_type == 'distribution':
            current_distribution = {'variables': block['variables'], 'properties':{},
                                    'probability':{}, 'default':None, 'table': None}
            for attr_type, attr in block['properties']:
                if attr_type == 'table':
                    current_distribution['table'] = attr
                elif attr_type == 'probability_entry':
                    current_distribution['probability'][tuple(attr[0])] = attr[1]
                elif attr_type == 'property':
                    current_distribution['properties'].update(attr)
                elif attr_type == 'default':
                    current_distribution['default'] = attr
            if len(current_distribution['probability']) == 0:
                current_distribution['probability'] = None
            p[0]['distributions'].append(current_distribution)

def p_error(p):
    print('SYNTAX ERROR:', p.type, p.value)

start = 'compilation_unit'

parser = yacc.yacc()

with open('example.bif', 'r') as f:
    s = f.read()

result = parser.parse(s)
for variable in result['variables']:
    print('variable {0}:'.format(variable['name']))
    print('properties: ', variable['properties'])
    print('n_values: ', variable['n_values'])
    print('values: ', variable['values_list'])
    print()

for distr in result['distributions']:
    print('distribution over {0}:'.format(distr['variables']))
    print('default: ', distr['default'])
    print('table: ', distr['table'])
    print('prob: ', distr['probability'])
    print('properties: ', distr['properties'])
    print()