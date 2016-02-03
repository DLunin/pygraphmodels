import ply.lex as lex

literals = ['{', '}', ';', '[', ']', '(', ')']

t_ignore = " \n\t\r|,"

tokens = (
    'DECIMAL_LITERAL',
    'FLOATING_POINT_LITERAL',
    'NETWORK',
    'PROPERTY',
    'PROBABILITY',
    'DEFAULTVALUE',
    'TABLEVALUES',
    'VARIABLE',
    'VARIABLETYPE',
    'DISCRETE',
    'WORD',
)


# A regular expression rule with some action code
def t_DECIMAL_LITERAL(t):
    r'\d+(?!\.)'
    t.value = int(t.value)
    return t


def t_FLOATING_POINT_LITERAL(t):
    r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
    t.value = float(t.value)
    return t


def t_NETWORK(t):
    r'network'
    return t


def t_PROPERTY(t):
    r'property'
    return t


def t_PROBABILITY(t):
    r'probability'
    return t


def t_DEFAULTVALUE(t):
    r'default'
    return t


def t_TABLEVALUES(t):
    r'table'
    return t


def t_VARIABLE(t):
    r'variable'
    return t


def t_VARIABLETYPE(t):
    r'type'
    return t


def t_DISCRETE(t):
    r'discrete'
    return t


def t_WORD(t):
    r'[^; {}()\[\],|]+'
    return t


def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


# Build the lexer
lexer = lex.lex()
