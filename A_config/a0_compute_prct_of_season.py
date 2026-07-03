import numpy as np
'''
Given sos and eos, it computes % to use in congig file for each predictions (every early-month form month 1 to the end
'''
### USER PART
# SOS and EOS in dek
sos = 7
eos = 21
### END OF USER PART

# The following is copied from a10
sosMonth = int(np.ceil(sos / 3)) # note ceil makes it correct, think about it
print('SOS, SosMonth', sos, sosMonth)
# take all the month where EOS fall in
eosMonth = int(np.ceil(eos / 3))  # same here
print('EOS, EosMonth', eos, eosMonth)
if sosMonth < eosMonth:
    real_months = list(range(int(sosMonth), int(eosMonth + 1)))
    plantingYearDelta = 0
else:
    real_months = list(range(int(sosMonth), 12 + 1)) + list(range(1, int(eosMonth) + 1))
    plantingYearDelta = -1
print('Calendar months of season')
print(real_months)

passed_months = list(range(0, len(real_months)))
print('Passed month at the beginning of each calendar month')
print(passed_months)
prct_of_season = np.ceil(np.array(passed_months) / len(real_months) * 100)
print('% of season at the beginning of each calendar month')
print(prct_of_season)

non_zero_prct_of_season = prct_of_season[1:]
print('non zero % of season at the beginning of each calendar month')
print(non_zero_prct_of_season)
# Check with a10 computation
print()
print('Check with a10')
forecastingPrct =non_zero_prct_of_season
id_months = np.array(range(1,len(real_months)+1))
# Now, no matter if 100 % forecast is used, store the last month in season to be used to limit use of Seasonal Forecast,
# if last month in season is 7, when prediction at 3, we should not use all the 7 months of SF
eosMonthInSeason = int(id_months[-1])
print('id_months')
print(id_months)
print('eosMonthInSeason')
print(eosMonthInSeason)
prct_months = id_months/len(id_months)*100
forecastingMonths = []
forecastingCalendarMonths = []
for prct in forecastingPrct:
    forecastingMonths = forecastingMonths + list([int(id_months[np.argmin(np.abs(prct_months-float(prct)))])])
    forecastingCalendarMonths =forecastingCalendarMonths + list([int(real_months[np.argmin(np.abs(prct_months-float(prct)))])])

print('a10 forecastingMonths')
print(forecastingMonths)
print('a10 forecastingCalendarMonths')
print(forecastingCalendarMonths)