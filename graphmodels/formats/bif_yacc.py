import ply.yacc as yacc
from pygraphmodels.graphmodels import DGM, TableFactor

# Get the token map from the lexer.  This is required.
from pygraphmodels.graphmodels.formats.bif_lex import tokens

#
# def p_number(p):
#     'expression : FLOATING_POINT_LITERAL'
#     p[0] = p[1]

def p_string(p):
    '''string : WORD
              | WORD string'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = [p[1]] + p[2]

def p_value(p):
    '''value : WORD
             | FLOATING_POINT_LITERAL
             | DECIMAL_LITERAL'''
    p[0] = p[1]

def p_list(p):
    '''list : value
            | value list'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = [p[1]] + p[2]


def p_property_string(p):
    "property_string : PROPERTY WORD string ';'"
    p[0] = 'property', {p[2]: ' '.join(p[3])}

def p_property_list(p):
    """property_list : property_string
                     | property_string property_list"""
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = [p[1]] + p[2]

def p_network_declaration(p):
    "network_declaration : NETWORK WORD network_content"
    p[0] = {'name': p[2], 'properties': p[3]}

def p_network_content(p):
    """network_content : '{' property_list '}'
                       | '{' '}'"""
    p[0] = {}
    if len(p) == 4:
        for prop in p[2]:
            p[0].update(prop)

def p_probability_variables_list(p):
    """probability_variables_list : '(' list ')'"""
    p[0] = p[2]

def p_probability_values_list(p):
    """probability_values_list : '(' list ')'"""
    p[0] = p[2]

def p_floating_point_list(p):
    '''floating_point_list : FLOATING_POINT_LITERAL
              | FLOATING_POINT_LITERAL floating_point_list'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = [p[1]] + p[2]

def p_probability_entry(p):
    """probability_entry : probability_values_list floating_point_list ';'"""
    p[0] = 'prob', p[1], p[2]

def p_probability_default_entry(p):
    """probability_default_entry : DEFAULTVALUE floating_point_list ';'"""
    p[0] = 'default', p[2]

def p_probability_table(p):
    """probability_table : TABLEVALUES floating_point_list ';'"""
    p[0] = 'table', p[2]

def p_probability_content(p):
    """probability_content : property_string probability_content
                           | probability_entry probability_content
                           | probability_default_entry probability_content
                           | probability_table probability_content
                           | """
    if len(p) == 1:
        p[0] = []
    else:
        p[0] = [p[1]] + p[2]

def p_probability_declaration(p):
    """probability_declaration : PROBABILITY probability_variables_list '{' probability_content '}'"""
    p[0] = { 'variables': p[2], 'properties': p[4] }


def p_variable_discrete(p):
    """variable_discrete : VARIABLETYPE DISCRETE '[' DECIMAL_LITERAL ']' '{' list '}' ';' """
    p[0] = { 'n_values': p[4], 'values_list': p[7] }

def p_variable_content(p):
    """variable_content : property_string variable_content
                        | variable_discrete variable_content
                        | """
    if len(p) == 1:
        p[0] = []
    else:
        p[0] = [p[1]] + p[2]


def p_variable_declaration(p):
    """variable_declaration : VARIABLE WORD '{' variable_content '}' """
    p[0] = 'variable', p[2], p[4]

def p_blocks_list(p):
    """blocks_list : variable_declaration blocks_list
                   | probability_declaration blocks_list
                   | """
    if len(p) == 1:
        p[0] = []
    else:
        p[0] = [p[1]] + p[2]

def p_compilation_unit(p):
    """compilation_unit : network_declaration blocks_list"""
    p[0] = [p[1]] + p[2]

def p_error(p):
    print('SYNTAX ERROR:', p.type, p.value)

start = 'compilation_unit'

parser = yacc.yacc()

with open('example.bif', 'r') as f:
    s = f.read()

result = parser.parse(s)
print(result)