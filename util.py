# Frequency of bad actions
def sumOf(dataTable, col):
    return float(sum(dataTable[col]))

def sumOfActions (dataTable):
    legal = sumOf(dataTable, 'action-legal')
    print "Legal actions: \t{}".format(legal)
    min_v = sumOf(dataTable, 'action-minor_violation')
    print "Minor violations: {}".format(min_v)
    maj_v = sumOf(dataTable, 'action-major_violation')
    print "Major violations: {}".format(maj_v)
    min_a = sumOf(dataTable, 'action-minor_accident')
    print "Minor accident: \t{}".format(min_a)
    maj_a = sumOf(dataTable, 'action-major_accident')
    print "Legal actions: \t{}".format(maj_a)
    sumAct = legal + min_v + maj_v + min_a + maj_a
    return {'action': ['legal', 'minor_violation', 'major_violation', 'minor_accident', 'major_accident'],'count':[legal, min_v, maj_v, min_a, maj_a]}

def getAvgActions (action_data):
    dataTable = sumOfActions(action_data)
    count = sumOf(dataTable, 'count')
    print
    print "Agent makes legal actions-\t {:,.2f}% of the time".format((dataTable['count'][0]/count)*100)
    print "Agent makes minor violations- {:,.2f}% of the time".format((dataTable['count'][1]/count)*100)
    print "Agent makes major violations- {:,.2f}% of the time".format((dataTable['count'][2]/count)*100)
    print "Agent has a minor accident-\t {:,.2f}% of the time".format((dataTable['count'][3]/count)*100)
    print "Agent has a major accident-\t {:,.2f}% of the time".format((dataTable['count'][4]/count)*100)