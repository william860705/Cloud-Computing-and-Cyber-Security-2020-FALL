#!/usr/bin/env python
"""mapper.py"""

import sys

month_to_num = {
    'Jan':'01', 'Feb':'02', 'Mar':'03',
    'Apr':'04', 'May':'05', 'Jun':'06',
    'Jul':'07', 'Aug':'08', 'Sep':'09',
    'Oct':'10', 'Nov':'11', 'Dec':'12',
}
# input comes from STDIN (standard input)
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    
    # split the line into words
    words = line.split()

    for word in words:
        # find [] in word
        if word[0]=='[' :
            date = word[1:12].split('/')
            hr = word[12:].split(':')[1]
            msg = '%s-%s-%s T %s:00:00.000' % \
            (date[2], month_to_num[date[1]], date[0], hr)

            print '%s\t%s' % (msg, 1)

