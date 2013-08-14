#!/usr/bin/env python

from datetime import date
import xlrd
from pylab import *

book=xlrd.open_workbook(filename='../Data/indian_wpi.xls')
sh=book.sheet_by_index(0)


# Create dict to match column headings to column numbers
Columns=dict(zip([s.strip() for s in sh.row_values(0)],range(sh.ncols)))
Rows=dict(zip([s.strip() for s in sh.col_values(0)],range(sh.nrows)))

# Get dates into proper format
dates=sh.row_values(Rows['COMM_NAME'])[Columns['INDX092005']:Columns['INDX092010']]

dates=[date(int(d[6:]),int(d[4:6]),1).toordinal() for d in dates]

# Get series on food price increases
y=sh.row_values(Rows['(A)  FOOD ARTICLES'])[Columns['INDX092005']:Columns['INDX092010']]

hold(True)

wpi=array(sh.row_values(Rows['ALL COMMODITIES'])[Columns['INDX092005']:Columns['INDX092010']])

rice=array(sh.row_values(Rows['Rice'])[Columns['INDX092005']:Columns['INDX092010']])/wpi

food=sh.row_values(Rows['(A)  FOOD ARTICLES'])[Columns['INDX092005']:Columns['INDX092010']]/wpi
cereals=sh.row_values(Rows['a1. CEREALS'])[Columns['INDX092005']:Columns['INDX092010']]/wpi

pulses=sh.row_values(Rows['a2. PULSES'])[Columns['INDX092005']:Columns['INDX092010']]/wpi

spices=sh.row_values(Rows['e.  CONDIMENTS & SPICES'])[Columns['INDX092005']:Columns['INDX092010']]/wpi

coconuts=sh.row_values(Rows['Coconut(Fresh)'])[Columns['INDX092005']:Columns['INDX092010']]/wpi 
 
plot_date(dates,food,'b-')

plot_date(dates,cereals,'g-')

plot_date(dates,pulses,'r-')

show()


