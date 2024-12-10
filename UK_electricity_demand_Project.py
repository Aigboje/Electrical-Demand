#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install kagglehub


# In[5]:


# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
albertovidalrod_electricity_consumption_uk_20092022_path = kagglehub.dataset_download('albertovidalrod/electricity-consumption-uk-20092022')

print('Data source import complete.')


# In[6]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[7]:


df = pd.read_csv("C:\\Users\\PC\\Desktop\\historic_demand_2009_2024_noNaN.csv",parse_dates=[1])
# Change column names to lower case and drop id (row number)
df.columns = df.columns.str.lower()

# Renaming columns
df.rename(columns = {'settlement_date': 'Date',
                    'settlement_period': 'Hourly_period',
                    'nd': 'National_demand',
                    'tsd': 'Total_demand'}, inplace=True)
()


# In[8]:


df.head()


# In[9]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from matplotlib import rcParams

import warnings
warnings.filterwarnings('ignore')


# In[12]:


pip install skforecast


# In[13]:


from skforecast.plot import set_dark_theme
set_dark_theme()


# In[14]:


df.head()


# In[15]:


df.sort_values(
    by=['Date', 'Hourly_period'], inplace=True, ignore_index=True
)


# In[16]:


df.info()


# In[17]:


df.describe()


# In[18]:


df.isna().sum()


# In[19]:


df.east_west_flow.value_counts()


# In[20]:


df.nemo_flow.value_counts()


# In[21]:


col = ['National_demand',
       'Total_demand', 'england_wales_demand', 'embedded_wind_generation',
       'embedded_wind_capacity', 'embedded_solar_generation',
       'embedded_solar_capacity', 'non_bm_stor', 'pump_storage_pumping',
       'ifa_flow', 'ifa2_flow', 'britned_flow', 'moyle_flow', 'east_west_flow',
       'nemo_flow', 'is_holiday']


# In[22]:


UK_df=df.groupby(['Date'])[['National_demand',
       'Total_demand', 'england_wales_demand', 'embedded_wind_generation',
       'embedded_wind_capacity', 'embedded_solar_generation',
       'embedded_solar_capacity', 'non_bm_stor', 'pump_storage_pumping',
       'ifa_flow', 'ifa2_flow', 'britned_flow', 'moyle_flow', 'east_west_flow',
       'nemo_flow', 'is_holiday']].mean()

UK_df.columns=col
UK_df.asfreq('D')
UK_df


# In[23]:


Demands = UK_df['2019-01-01':][['National_demand','Total_demand','england_wales_demand']]
Demands


# In[24]:


Demands.plot(figsize=(20,5),linewidth=1.2,cmap='cool',ylabel='MW',title='Electricity Demand')


# In[25]:


for i in Demands.columns:
  sns.histplot(data=Demands, x=i, bins=500, color="g")
  plt.show()


# In[26]:


null_days = df.loc[df["Total_demand"] == 0.0, "Date"].unique().tolist()

null_days_index = []

for day in null_days:
    null_days_index.append(df[df["Day"] == day].index.tolist())

null_days_index = [item for sublist in null_days_index for item in sublist]

df.drop(index=null_days_index, inplace=True)
df.reset_index(drop=True, inplace=True)


# In[27]:


import calendar
lab=['National ','Transmission System ','ENGLAND_WALES']
colours = ['darkturquoise','fuchsia','tomato']
# Ensure the index is a datetime index

if not isinstance(UK_df.index, pd.DatetimeIndex):
    UK_df.index = pd.to_datetime(UK_df.index)

# Now you can safely access the `.month` attribute
fig, axs = plt.subplots(3, 1, figsize=(30, 14), sharex=False)

for i, j, k in zip(UK_df.columns, range(0, 3), lab):
    sns.boxplot(data=UK_df, x=UK_df.index.month, y=i, ax=axs[j], palette='Set3', saturation=1)
    axs[j].set_xticklabels([z for z in calendar.month_abbr[1:]], rotation=30, fontsize=15)
    axs[j].set_title(f'Electricity Demand: {k}', fontsize=20, fontweight='bold')


# In[28]:


f,axs=plt.subplots(3,1,figsize=(20,12),sharex=False)
for i,j,l,k in zip(UK_df.columns,[0,1,2], colours,lab):
        sns.histplot(data=UK_df,bins=40,x=i,ax=axs[j],kde=True,stat='density',color=l)
        axs[j].set_title(f'Electricity Demand :{k}',fontsize=20,fontweight='bold')
plt.tight_layout()
plt.show()


# In[29]:


UK_df


# In[30]:


fig,axs=plt.subplots(3,1,figsize=(20,10))
ys=UK_df.columns       #i         #j       #k                                  #l
for i,j,l,k in zip(UK_df.columns,[0,1,2], colours,lab):
    sns.lineplot(data=UK_df,x=UK_df.index,y=i,ax=axs[j],label=k,color=l,linewidth=1.6)
    axs[j].set_title(f'  Electricity Demand :  {k}',fontsize=18)
    axs[j].set_ylabel('Mega Watts')
plt.tight_layout()
plt.title('Electricity Demand Line Chart')
plt.show()


# In[31]:


pip install holidays


# In[32]:


import holidays

# Compare England's and Wales' bank holiday
bank_holiday_england = holidays.UK(
    subdiv="England", years=range(2009, 2024), observed=True
).items()
bank_holiday_wales = holidays.UK(
    subdiv="Wales", years=range(2009, 2024), observed=True
).items()

print(bank_holiday_england == bank_holiday_wales)


# In[33]:


df.head()


# In[34]:


# Create empty lists to store data
holiday_names = []
holiday_dates = []
holiday_names_observed = []
holiday_dates_observed = []

for date, name in sorted(bank_holiday_england):
    holiday_dates.append(date)
    holiday_names.append(name)
    # Pop the previous value as observed bank holidays takes place later
    if "Observed" in name:
        holiday_dates_observed.pop()
        holiday_names_observed.pop()

    holiday_names_observed.append(name)
    holiday_dates_observed.append(np.datetime64(date))

holiday_names_observed[:10]


# In[35]:


df_main = df.copy(deep = True)
# df_main.drop(columns = ['period_hour'], inplace = True)
df_main.head()


# In[36]:


df_main["is_holiday"] = df_main["Date"].apply(
    lambda x: pd.to_datetime(x) in holiday_dates_observed
)

df_main["is_holiday"] = df_main["is_holiday"].astype(int)

df_main[df_main["is_holiday"] == 0].sample(10)


# In[37]:


# Set date as the index and turn into datetime type
df_plot = df_main.copy()
df_plot = df_plot.set_index("Date")
df_plot.index = pd.to_datetime(df_plot.index)

fig, ax = plt.subplots(figsize=(15, 5))
df_plot["Total_demand"].plot(
    style=".", ax=ax, title="Trasnmission System Demand", label="Timeries data"
)
(df_plot.query("is_holiday == 1")["is_holiday"] * 33000).plot(
    style=".", ax=ax, label="Bank holiday"
)
ax.legend();


# In[38]:


df_plot.head()


# In[39]:


df_plot.columns


# In[40]:


sample = df_plot['2022-01-01':][['National_demand','Total_demand','england_wales_demand', 'embedded_wind_generation',
       'embedded_wind_capacity', 'embedded_solar_generation',
       'embedded_solar_capacity', 'non_bm_stor', 'pump_storage_pumping',
       'ifa_flow', 'ifa2_flow', 'britned_flow', 'moyle_flow', 'east_west_flow',
       'nemo_flow', 'is_holiday']]
sample.head()


# In[41]:


sample['2024-01-01':][['National_demand','Total_demand','england_wales_demand']].plot(figsize=(20,5),linewidth=3,
                                        title='Electricity Demand 2024',cmap='brg',ylabel='MW')


# In[42]:


Demands.loc[(Demands.index > "01-01-2019")]['Total_demand']


# In[43]:


# Convert string dates to datetime objects
start_date_2019 = pd.to_datetime("2019-01-01")
end_date_2019 = pd.to_datetime("2019-12-01")
start_date_2023 = pd.to_datetime("2023-01-01")
end_date_2023 = pd.to_datetime("2023-12-01")

# Ensure Demands.index is in datetime format
Demands.index = pd.to_datetime(Demands.index)

fig, ax = plt.subplots(figsize=(15, 7))

# Filter data using datetime objects
demand_2019 = Demands.loc[(Demands.index > start_date_2019) & (Demands.index < end_date_2019)]["Total_demand"]
demand_2023 = Demands.loc[(Demands.index > start_date_2023) & (Demands.index < end_date_2023)]["Total_demand"]

ax.plot(range(len(demand_2019)), demand_2019, "o", label="2016")
ax.plot(range(len(demand_2023)), demand_2023, "o", alpha=0.5, label="2023")  # Changed label to 2023


ax.set_xlabel("Data sample")
ax.set_ylabel("Electricity demand (MW)")
ax.legend(loc="best")
ax.set_title("Demand comparison - 2019 and 2023")
plt.show()


# In[44]:


correlation_matrix = sample.corr()

# Heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="cubehelix")
plt.title("Correlation Matrix")
plt.show()


# Decomposition

# In[45]:


pip install statsmodels


# In[46]:


from statsmodels.tsa.seasonal import seasonal_decompose



# seasonal decompose - visual test for stationarity
def seas_decomp(dataframe,p):
    sdf=pd.DataFrame()
    cols=['Trend(t)','Seasonal(S)','Residuals(e)','observed(o)']

    for i in ['additive','multiplicative']:
            if i=='additive':
                result=seasonal_decompose(x=dataframe,model=i,period=p)
                li=[result.trend,result.seasonal,result.resid,result.observed]
                for j,k in zip(cols,li):
                    sdf[f'+_{j}']=k
            elif i=='multiplicative':
                result=seasonal_decompose(x=dataframe,model=i,period=p)
                li=[result.trend,result.seasonal,result.resid,result.observed]
                for j,k in zip(cols,li):
                    sdf[f'*_{j}']=k

    return sdf

National_demand = seas_decomp(sample['National_demand'],90)
Total_demand = seas_decomp(sample['Total_demand'],90)

# Add a constant to make all values positive (if values are near zero or negative)
sample['england_wales_demand'] += abs(sample['england_wales_demand'].min()) + 1

EW_demand=seas_decomp(sample['england_wales_demand'],90)


# In[47]:


def seas_decop_plot(dataframe,name):
    #col=['darkorange','yellowgreen','darksalmon','darkmagenta']
    lab=['National ','Transmission System ','ENGLAND_WALES']
    co=['crimson','orangered','k','darkblue','crimson','orangered','k','darkblue']#'darkred'
    fig,axs=plt.subplots(8,1,figsize=(100,80))
    print(name)
    for i,j,k in zip(dataframe.columns,range(0,8),co):
        sns.lineplot(data=dataframe, x=dataframe.index,y=i,ax=axs[j],color=k)
        axs[j].set_title(f'{name}: {i}',fontsize=80,fontweight='bold')
        axs[j].xaxis.set_tick_params(labelsize=30)
        axs[j].yaxis.set_tick_params(labelsize=30)
        #ax.xaxis.set_tick_params(labelsize=20)

    plt.tight_layout()
    plt.show()

for i,j in zip((National_demand,Total_demand,EW_demand),['National ','Transmission System ','ENGLAND_WALES']):

    seas_decop_plot(i,j)
    print('')
    print('*'*162)


# In[48]:


print(sample.index.duplicated().sum())


# In[49]:


sample.tail()


# STATISTICAL TEST FOR STATIONARITY-TIME SERIES

# In[50]:


from statsmodels.tsa.stattools import adfuller,kpss

#Stationarity Tests

def stationary_test(dataframe,):

    print('AUGUMENTED-DICKEY-FULLER-TEST:')
    print('')

#hypothesis statement
    H0_adf='Data is not Stationary in nature, ie: unit root = 1'
    H1_adf='Data is Stationary in nature, ie: unitroot < 1, Further Procced for modeling!'

    for i in dataframe.columns:
        print('')
        print(i,':')
        print('')
        c=adfuller(dataframe[i])
        print('')
        #decision based on p value
        if c[1]>0.05:
                print(f"Decision: {H0_adf}")
        else:
                print(f"Decision: {H1_adf}")
        print('')
        for j,k in zip(['ADF test statistic','P value','No of lags used','No of Observations used',
                 'Information Criterion value'],c):
            print(f"{j}: {k}")

    print('')
    print('*'*160)
    print('')
    print('KIWATKOWSKI-PHILIPS-SCHMIDT-SHIN (KPSS) TEST')
    print('')

    #hypothesis statement
    h0_kp='Data is Stationary in nature, Futher steps for modelling is approved'
    h1_kp='Data is not Stationary in nature'

    for i in dataframe.columns:
        print('')
        print(i,':')
        k=kpss(dataframe[i],regression='c')
        print('')
        if k[1]>0.05:
            print('Decision',h0_kp)
        else:
            print('Decision',h1_kp)
        print('')
        for a,b in zip(['KPSS test statistic','P value','No of lags used',
                  'Critical values in %'],k):
            print(f"{a}: {b}")


stationary_test(sample)


# Because our data appears to be steady according to the ADF test. However, the KPSS test indicates that it is not stationary. Let us apply some p periods of differencing on our data in order to produce stationarity.

# In[51]:


def Induced_stationarity(dataframe,p):
    temp_df=pd.DataFrame()
    for i in dataframe.columns:
        temp_df[f'{i}']=dataframe[i].diff(periods=p,)
    return temp_df

Induced = Induced_stationarity(sample,30)
Induced.dropna(axis=0,inplace=True)
print(Induced.isnull().sum())
Induced.head()


# In[52]:


f,a=plt.subplots(3,1,figsize=(30,15))
for i,j,k,l in zip(Induced.columns,range(0,3),['darkorchid','saddlebrown','lawngreen'],lab):
    sns.lineplot(data=Induced,x=Induced.index,y=i,ax=a[j],color=k,linewidth=3)
    a[j].set_title(f'Demands: {l}',fontweight='bold',fontsize=20)
plt.tight_layout()


# In[53]:


stationary_test(Induced)


# In[54]:


National_demand=Induced['National_demand'].to_frame()

Transmission_Demand=Induced['Total_demand'].to_frame()

England_Wales_Demand=Induced['england_wales_demand'].to_frame()


# In[55]:


get_ipython().system('pip install sktime')


# In[56]:


from sktime.forecasting.model_selection import temporal_train_test_split

Nd_train,Nd_test=temporal_train_test_split(National_demand,test_size=1000)
Tsd_train,Tsd_test=temporal_train_test_split(Transmission_Demand,test_size=1000)
Ewd_train,Ewd_test=temporal_train_test_split(England_Wales_Demand,test_size=1000)


# In[57]:


Ewd_train.shape,Ewd_test.shape


# SMOOTHING TECHNIQUES

# In[58]:


from sktime.utils.plotting import plot_series
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error as mae
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


# In[59]:


# Simple exponential Smoothing function

def SES(train,test,ses_initialization,f_steps,s):

    min_mse = np.inf
    best_smooth_level = None
    best_df = None
    best_model = None
    for i in np.arange(0.01, 1, s):
        m = SimpleExpSmoothing(endog=train, initialization_method=ses_initialization).fit(smoothing_level=i)
        # forecast step
        f = m.forecast(steps=f_steps)

        # fitted values
        fttd = m.fittedvalues

        # dataframe creation
        dates = pd.date_range(start=train.index[-1], periods=f_steps, freq='B')
        ses_dataframe = pd.DataFrame({'date': dates, 'Forecasts': f})
        ses_dataframe = ses_dataframe.set_index('date')

        # metrics
        # Ensure both arrays have the same length for comparison
        # Use the minimum length to avoid errors
        min_len = min(len(test), len(ses_dataframe['Forecasts']))
        ms = mse(test.iloc[:min_len], ses_dataframe['Forecasts'].iloc[:min_len])
        ma = mape(test.iloc[:min_len], ses_dataframe['Forecasts'].iloc[:min_len])
        m1ae = mae(test.iloc[:min_len], ses_dataframe['Forecasts'].iloc[:min_len])

        if ms < min_mse:
            min_mse = ms
            best_smooth_level = i
            best_df = ses_dataframe
            best_model = m
    print(f"Best Smoothing level : {best_smooth_level}")   #plottings
    print(best_model.summary())
      #train-test-forecast plot
    fig,a=plt.subplots(3,1,figsize=(30,20))
    train.plot(c='limegreen',ax=a[0])
    test.plot(c='salmon',ax=a[0])
    best_df.plot(c='darkslateblue',ax=a[0])
    a[0].set_title(f"Simple Exponential Smoothing-{best_smooth_level}", fontweight='bold',fontsize=35)
    a[0].legend(['Train','Test',f'Forecast-{best_smooth_level}'],loc='best',fontsize='20')


    #fitted vs train plot
    train.plot(c='limegreen',ax=a[1])
    fttd.plot(ax=a[1],c='yellow')
    a[1].set_title(f"Fitted - Simple Exponential Smoothing-{best_smooth_level}", fontweight='bold',fontsize=35)
    a[1].legend(['Train',f'Fitted-{best_smooth_level}'],loc='best',fontsize='20')

    #ONLY FORECAST
    best_df.plot(ax=a[2],c='darkslateblue')
    test.plot(c='salmon',ax=a[2])
    a[2].set_title(f"Forecasted - Simple Exponential Smoothing-{best_smooth_level}", fontweight='bold',fontsize=35)
    a[2].legend([f'Forecast-{best_smooth_level}','Test'],loc='best',fontsize='20')


    plt.tight_layout()
    plt.show()


    for n,N in zip(['MSE','MAPE','MAE'],[min_mse,ma,m1ae]):
        print(f"{n} is {N}")

    return best_df


# In[60]:


sesnd=SES(Nd_train,Nd_test,'heuristic',365,0.02)


# CORRELATION PLOTS

# In[61]:


from sktime.utils.plotting import plot_correlations


def Corr(s,lag,sup_title,c): # give the input as series object
    plt.figure(figsize=(20,4))
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[c])
    plot_correlations(series=s, lags=lag,alpha=0.05,suptitle=sup_title,)
    plt.tight_layout()
    plt.show()


# In[62]:


Corr(Nd_train,70,'National Demand','tomato',)


# In[63]:


Corr(Tsd_train,70,'Transmission Demand','darkorchid')


# In[64]:


Corr(Ewd_train,70,'England-Wales Demand','firebrick')


# ### MODEL BUILDING

# #### ARIMA MODEL

# In[65]:


pip install pmdarima


# In[66]:


from pmdarima import auto_arima
import datetime as dt


# In[67]:


sample.head()


# In[68]:


sample.columns


# In[69]:


features = ['embedded_wind_generation', 'embedded_wind_capacity',
       'embedded_solar_generation', 'embedded_solar_capacity', 'non_bm_stor',
       'pump_storage_pumping', 'ifa_flow', 'ifa2_flow', 'britned_flow',
       'moyle_flow', 'east_west_flow', 'nemo_flow', 'is_holiday']


# In[70]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score


# NATIONAL DEMAND

# In[71]:


Nd_train = sample[sample.index < "2024"]
Nd_valid = sample[sample.index >= "2024"]


# In[72]:


model = auto_arima(Nd_train.National_demand, exogenous=Nd_train[features], trace=True, error_action="ignore", suppress_warnings=True)
model.fit(Nd_train.National_demand, exogenous=Nd_train[features])

Nd_forecast = model.predict(n_periods=len(Nd_valid), exogenous=Nd_valid[features])


# In[73]:


Nd_valid.reset_index(inplace = True)

Nd_forecast = Nd_forecast.to_frame(name="ND_forecast")
Nd_forecast.reset_index(inplace = True)
Nd_forecast.drop(columns = 'index', inplace = True)

ND_predictions = pd.concat([Nd_valid, Nd_forecast], axis=1)
ND_predictions.head()


# In[74]:


print("MSE of Auto ARIMAX:", mean_squared_error(ND_predictions.National_demand, ND_predictions.ND_forecast))
print("RMSE of Auto ARIMAX:", np.sqrt(mean_squared_error(ND_predictions.National_demand, ND_predictions.ND_forecast)))
print("MAE of Auto ARIMAX:", mean_absolute_error(ND_predictions.National_demand, ND_predictions.ND_forecast))
print("MAPE of Auto ARIMAX:", mean_absolute_percentage_error(ND_predictions.National_demand, ND_predictions.ND_forecast))
print("R2_score of Auto ARIMAX:", r2_score(ND_predictions.National_demand, ND_predictions.ND_forecast))


# In[75]:


ND_predictions[["National_demand", "ND_forecast"]].plot(figsize=(14, 7), color = ['red', 'black'])


# In[76]:


plt.figure(figsize = (15, 5))
plt.plot(Nd_train.index, Nd_train["National_demand"], label="Training set")
plt.plot(Nd_valid.index, Nd_valid["National_demand"], label="Test set")
plt.plot(Nd_valid.index, ND_predictions["ND_forecast"], label="Predictions")
plt.axvline(Nd_valid.index.min(), color="y", ls="--")
plt.legend(loc="center", bbox_to_anchor=(1.075, 0.5))

plt.title("Prediction on test set(National demand) - week")
plt.ylabel("National Energy Demand (MW)")
plt.xlabel("Date")


# TRANSMISSION DEMAND

# In[77]:


Td_train = sample[sample.index < "2024"]
Td_valid = sample[sample.index >= "2024"]


# In[78]:


model = auto_arima(Td_train.Total_demand, exogenous=Td_train[features], trace=True, error_action="ignore", suppress_warnings=True)
model.fit(Td_train.Total_demand, exogenous=Td_train[features])

Td_forecast = model.predict(n_periods=len(Td_valid), exogenous=Td_valid[features])


# In[79]:


Td_forecast.dtype


# In[80]:


Td_valid.reset_index(inplace = True)

Td_forecast = Td_forecast.to_frame(name="TD_forecast")
Td_forecast.reset_index(inplace = True)
Td_forecast.drop(columns = 'index', inplace = True)

TD_predictions = pd.concat([Td_valid, Td_forecast], axis=1)
TD_predictions.head()


# In[81]:


print("MSE of Auto ARIMAX(Transmission):", mean_squared_error(TD_predictions.Total_demand, TD_predictions.TD_forecast))
print("RMSE of Auto ARIMAX(Transmission):", np.sqrt(mean_squared_error(TD_predictions.Total_demand, TD_predictions.TD_forecast)))
print("MAE of Auto ARIMAX(Transmission):", mean_absolute_error(TD_predictions.Total_demand, TD_predictions.TD_forecast))
print("MAPE of Auto ARIMAX(Transmission):", mean_absolute_percentage_error(TD_predictions.Total_demand, TD_predictions.TD_forecast))
print("R2_score of Auto ARIMAX(Transmission):", r2_score(TD_predictions.Total_demand, TD_predictions.TD_forecast))


# In[82]:


TD_predictions[["Total_demand", "TD_forecast"]].plot(figsize=(14, 7), color = ['red', 'black'])


# ENGLAND_WALES DEMAND

# In[83]:


EW_train = sample[sample.index < "2024"]
EW_valid = sample[sample.index >= "2024"]


# In[84]:


model = auto_arima(EW_train.england_wales_demand, exogenous=EW_train[features], trace=True, error_action="ignore", suppress_warnings=True)
model.fit(EW_train.england_wales_demand, exogenous=EW_train[features])

EW_forecast = model.predict(n_periods=len(EW_valid), exogenous=EW_valid[features])


# In[85]:


EW_valid.reset_index(inplace = True)

EW_forecast = EW_forecast.to_frame(name="EW_forecast")
EW_forecast.reset_index(inplace = True)
EW_forecast.drop(columns = 'index', inplace = True)

EW_predictions = pd.concat([EW_valid, EW_forecast], axis=1)
EW_predictions.head()


# In[86]:


EW_predictions[["england_wales_demand", "EW_forecast"]]


# In[87]:


print("MSE of Auto ARIMAX(England_Wales demand):", mean_squared_error(EW_predictions.england_wales_demand, EW_predictions.EW_forecast))
print("RMSE of Auto ARIMAX(England_Wales demand):", np.sqrt(mean_squared_error(EW_predictions.england_wales_demand, EW_predictions.EW_forecast)))
print("MAE of Auto ARIMAX(England_Wales demand):", mean_absolute_error(EW_predictions.england_wales_demand, EW_predictions.EW_forecast))
print("MAPE of Auto ARIMAX(England_Wales demand):", mean_absolute_percentage_error(EW_predictions.england_wales_demand, EW_predictions.EW_forecast))
print("R2_score of Auto ARIMAX(England_Wales demand):", r2_score(EW_predictions.england_wales_demand, EW_predictions.EW_forecast))


# In[88]:


EW_predictions[["england_wales_demand", "EW_forecast"]].plot(figsize=(14, 7), color = ['red', 'black'])


# In[89]:


plt.figure(figsize = (15, 5))
plt.plot(Td_train.index, Td_train["Total_demand"], label="Training set")
plt.plot(Td_valid.index, Td_valid["Total_demand"], label="Test set")
plt.plot(Td_valid.index, TD_predictions["TD_forecast"], label="Predictions")
plt.axvline(Nd_valid.index.min(), color="y", ls="--")
plt.legend(loc="center", bbox_to_anchor=(1.075, 0.5))

plt.title("Prediction on test set(Total demand) - week")
plt.ylabel("Total Energy Demand (MW)")
plt.xlabel("Date")


# In[ ]:





# ## FACEBOOK PROPHET

# In[90]:


pip install prophet


# In[91]:


from prophet import Prophet


# In[92]:


Nd_train_fbp = sample[sample.index < "2024"]
Nd_valid_fbp = sample[sample.index >= "2024"]

Nd_train_fbp.reset_index(inplace = True)
Nd_valid_fbp.reset_index(inplace = True)


# In[93]:


model_fbp = Prophet()
for feature in features:
    model_fbp.add_regressor(feature)

model_fbp.fit(Nd_train_fbp[["Date", "National_demand"] + features].rename(columns={"Date": "ds", "National_demand": "y"}))

Ndp_forecast = model_fbp.predict(Nd_valid[["Date", "National_demand"] + features].rename(columns={"Date": "ds"}))


# In[94]:


Ndp_forecast


# In[95]:


Nd_valid_fbp["Forecast_Prophet"] = Ndp_forecast.yhat.values
Nd_valid_fbp.head()


# In[96]:


model_fbp.plot_components(Ndp_forecast)


# In[97]:


print("MSE of Prophet:", mean_squared_error(Nd_valid_fbp.National_demand, Nd_valid_fbp.Forecast_Prophet))
print("RMSE of Prophet:", np.sqrt(mean_squared_error(Nd_valid_fbp.National_demand, Nd_valid_fbp.Forecast_Prophet)))
print("MAE of Prophet:", mean_absolute_error(Nd_valid_fbp.National_demand, Nd_valid_fbp.Forecast_Prophet))
print("MAPE of Prophet:", mean_absolute_percentage_error(Nd_valid_fbp.National_demand, Nd_valid_fbp.Forecast_Prophet))
print("R2score of Prophet:", r2_score(Nd_valid_fbp.National_demand, Nd_valid_fbp.Forecast_Prophet))


# In[98]:


Nd_valid_fbp[["National_demand", "Forecast_Prophet"]].plot(figsize=(14, 7), color = ['b', 'm'])


# In[99]:


plt.figure(figsize = (15, 5))
plt.plot(Nd_train_fbp["Date"], Nd_train_fbp["National_demand"], label="Training set")
plt.plot(Nd_valid_fbp["Date"], Nd_valid_fbp["National_demand"], label="Test set")
plt.plot(Nd_valid_fbp["Date"], Nd_valid_fbp["Forecast_Prophet"], label="Prediction")
plt.axvline(Nd_valid_fbp["Date"].min(), color="y", ls="--")
plt.legend(loc="center", bbox_to_anchor=(1.075, 0.5))

plt.title("Prediction on test set(National demand) - week")
plt.ylabel("Energy Demand (MW)")
plt.xlabel("Date")


# **Transmission Demand**

# In[100]:


Td_train_fbp = sample[sample.index < "2024"]
Td_valid_fbp = sample[sample.index >= "2024"]

Td_train_fbp.reset_index(inplace = True)
Td_valid_fbp.reset_index(inplace = True)


# In[101]:


model_fbpt = Prophet()
for feature in features:
    model_fbpt.add_regressor(feature)
model_fbpt.fit(Td_train_fbp[["Date", "Total_demand"] + features].rename(columns={"Date": "ds", "Total_demand": "y"}))

Tdp_forecast = model_fbpt.predict(Td_valid_fbp[["Date", "Total_demand"] + features].rename(columns={"Date": "ds"}))
Td_valid_fbp["Td_Forecast_Prophet"] = Tdp_forecast.yhat.values


# In[102]:


Tdp_forecast.head()


# In[103]:


model_fbp.plot_components(Tdp_forecast)


# In[104]:


print("MSE of Prophet for transmission:", mean_squared_error(Td_valid_fbp.Total_demand, Td_valid_fbp.Td_Forecast_Prophet))
print("RMSE of Prophet for transmission:", np.sqrt(mean_squared_error(Td_valid_fbp.Total_demand, Td_valid_fbp.Td_Forecast_Prophet)))
print("MAE of Prophet for transmission:", mean_absolute_error(Td_valid_fbp.Total_demand, Td_valid_fbp.Td_Forecast_Prophet))
print("MAPE of Prophet for transmission:", mean_absolute_percentage_error(Td_valid_fbp.Total_demand, Td_valid_fbp.Td_Forecast_Prophet))
print("R2_score of Prophet for transmission:", r2_score(Td_valid_fbp.Total_demand, Td_valid_fbp.Td_Forecast_Prophet))


# In[105]:


Td_valid_fbp[["Total_demand", "Td_Forecast_Prophet"]].plot(figsize=(14, 7), color = ['b', 'm'])


# In[106]:


plt.figure(figsize = (15, 5))
plt.plot(Td_train_fbp["Date"], Td_train_fbp["Total_demand"], label="Training set")
plt.plot(Td_valid_fbp["Date"], Td_valid_fbp["Total_demand"], label="Test set")
plt.plot(Td_valid_fbp["Date"], Td_valid_fbp["Td_Forecast_Prophet"], label="Predictions")
plt.axvline(Td_valid_fbp["Date"].min(), color="y", ls="--")
plt.legend(loc="center", bbox_to_anchor=(1.075, 0.5))

plt.title("Prediction on test set(Transmission demand) - week")
plt.ylabel("Energy Demand (MW)")
plt.xlabel("Date")


# **England-Wales demand**

# In[107]:


Ew_train_fbp = sample[sample.index < "2024"]
Ew_valid_fbp = sample[sample.index >= "2024"]

Ew_train_fbp.reset_index(inplace = True)
Ew_valid_fbp.reset_index(inplace = True)


# In[108]:


model_fbpEW = Prophet()
for feature in features:
    model_fbpEW.add_regressor(feature)


model_fbpEW.fit(Ew_train_fbp[["Date", "england_wales_demand"] + features].rename(columns={"Date": "ds", "england_wales_demand": "y"}))

Ewfp_forecast = model_fbpEW.predict(Ew_valid_fbp[["Date", "england_wales_demand"] + features].rename(columns={"Date": "ds"}))
Ew_valid_fbp["EW_Forecast_Prophet"] = Ewfp_forecast.yhat.values


# In[109]:


Ew_train_fbp.head()


# In[110]:


Ewfp_forecast.head()


# In[111]:


model_fbp.plot_components(Ewfp_forecast)


# In[112]:


print("MSE of Prophet for england_wales_demand:", mean_squared_error(Ew_valid_fbp.england_wales_demand, Ew_valid_fbp.EW_Forecast_Prophet))
print("RMSE of Prophet for england_wales_demand:", np.sqrt(mean_squared_error(Ew_valid_fbp.england_wales_demand, Ew_valid_fbp.EW_Forecast_Prophet)))
print("MAE of Prophet for england_wales_demand:", mean_absolute_error(Ew_valid_fbp.england_wales_demand, Ew_valid_fbp.EW_Forecast_Prophet))
print("MAPE of Prophet for england_wales_demand:", mean_absolute_percentage_error(Ew_valid_fbp.england_wales_demand, Ew_valid_fbp.EW_Forecast_Prophet))


# In[113]:


Ew_valid_fbp[["england_wales_demand", "EW_Forecast_Prophet"]].plot(figsize=(14, 7), color = ['b', 'm'])


# In[114]:


plt.figure(figsize = (15, 10))
plt.plot(Ew_train_fbp["Date"], Ew_train_fbp["england_wales_demand"], label="Training set")
plt.plot(Ew_valid_fbp["Date"], Ew_valid_fbp["england_wales_demand"], label="Test set")
plt.plot(Ew_valid_fbp["Date"], Ew_valid_fbp["EW_Forecast_Prophet"], label="Predictions")
plt.axvline(Ew_valid_fbp["Date"].min(), color="y", ls="--")
plt.legend(loc="center", bbox_to_anchor=(1.075, 0.5))

plt.title("Prediction on test set - week")
plt.ylabel("Energy Demand (MW)")
plt.xlabel("Date")


# ### SarimaX

# In[115]:


from skforecast.plot import set_dark_theme
set_dark_theme()
from skforecast.sarimax import Sarimax
from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[116]:


target = Nd_train['National_demand']
exog = Nd_train.drop(columns = ['National_demand'])
# Assign a frequency to the index (e.g., daily data)
target = target.asfreq('D')
exog = exog.asfreq('D')


# In[117]:


# ARIMA model with skforecast.Sarimax
# ==============================================================================
warnings.filterwarnings("ignore", category=UserWarning, message='Non-invertible|Non-stationary')
model = Sarimax(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model.fit(y = target, exog = exog)
model.summary()
warnings.filterwarnings("default")


# In[118]:


Nd_train.shape, Nd_valid.shape


# In[127]:


Nd_valid.head()


# In[128]:


Nd_valid.shape
# Nd_valid.set_index('Date', inplace = True)


# In[131]:


test = Nd_valid['National_demand']
exog_test = Nd_valid.drop(columns = ['National_demand'])
# Assign a frequency to the index (e.g., daily data)
test = test.asfreq('D')
exog_test = exog_test.asfreq('D')


# In[132]:


# Prediction
# ==============================================================================

NdX_predictions = model.predict(steps=len(test), exog=exog_test)


NdX_predictions.columns = ['Nd_forecast']
display(NdX_predictions.head())


# In[133]:


print("MSE of Sarimax for National_demand:", mean_squared_error(test, NdX_predictions.Nd_forecast))
print("RMSE of Sarimax for National_demand:", np.sqrt(mean_squared_error(test, NdX_predictions.Nd_forecast)))
print("MAE of Sarimax for National_demand:", mean_absolute_error(test, NdX_predictions.Nd_forecast))
print("MAPE of Sarimax for National_demand:", mean_absolute_percentage_error(test, NdX_predictions.Nd_forecast))
print("R2_score of Sarimax for National_demand:", r2_score(test, NdX_predictions.Nd_forecast))


# In[134]:


# Plot predictions, 
# ==============================================================================
set_dark_theme()

fig, ax = plt.subplots(figsize=(15, 10))
Nd_train.National_demand.plot(ax=ax, label='train', color = 'r')
Nd_valid.National_demand.plot(ax=ax, label='test', color = 'g')
NdX_predictions.plot(ax=ax, label='skforecast', color = 'b')
ax.set_title('Predictions with SARIMA models')
# ax.legend();


# #### Transmission demand

# In[135]:


Td_trainX = Td_train.copy(deep = True)
Td_validX = Td_valid.copy(deep = True)


# In[136]:


Td_validX.set_index('Date', inplace = True)
Td_validX.head()


# In[137]:


Td_target = Td_trainX['Total_demand']
Td_exog = Td_trainX.drop(columns = ['Total_demand'])
# Assign a frequency to the index (e.g., daily data)
Td_target = Td_target.asfreq('D')
Td_exog = Td_exog.asfreq('D')


# In[138]:


# ARIMA model with skforecast.Sarimax
# ==============================================================================
warnings.filterwarnings("ignore", category=UserWarning, message='Non-invertible|Non-stationary')
model = Sarimax(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model.fit(y = Td_target, exog = Td_exog)
model.summary()
warnings.filterwarnings("default")


# In[139]:


Td_test = Td_validX['Total_demand']
Td_exog_test = Td_validX.drop(columns = ['Total_demand'])
# Assign a frequency to the index (e.g., daily data)
Td_test = Td_test.asfreq('D')
Td_exog_test = Td_exog_test.asfreq('D')


# In[140]:


# Prediction
# ==============================================================================

TdX_predictions = model.predict(steps=len(Td_test), exog=Td_exog_test)


TdX_predictions.columns = ['Td_forecast']
display(TdX_predictions.head())


# In[141]:


# Plot predictions, 
# ==============================================================================
set_dark_theme()

fig, ax = plt.subplots(figsize=(15, 10))
Td_trainX.Total_demand.plot(ax=ax, label='train', color = 'r')
Td_validX.Total_demand.plot(ax=ax, label='test', color = 'g')
TdX_predictions.plot(ax=ax, label='skforecast', color = 'b')
ax.set_title('Predictions with SARIMA models')


# In[142]:


print("MSE of Sarimax for Total_demand:", mean_squared_error(Td_test, TdX_predictions.Td_forecast))
print("RMSE of Sarimax for Total_demand:", np.sqrt(mean_squared_error(Td_test, TdX_predictions.Td_forecast)))
print("MAE of Sarimax for Total_demand:", mean_absolute_error(Td_test, TdX_predictions.Td_forecast))
print("MAPE of Sarimax for Total_demand:", mean_absolute_percentage_error(Td_test, TdX_predictions.Td_forecast))
print("R2_score of Sarimax for Total_demand:", r2_score(Td_test, TdX_predictions.Td_forecast))


# In[ ]:





# #### England-Wales demand

# In[149]:


Ew_trainX = EW_train.copy(deep = True)
Ew_validX = EW_valid.copy(deep = True)


# In[155]:


Ew_validX.shape


# In[154]:


Ew_validX.set_index('Date', inplace = True)


# In[156]:


Ew_target = Ew_trainX['england_wales_demand']
Ew_exog = Ew_trainX.drop(columns = ['england_wales_demand'])
# Assign a frequency to the index (e.g., daily data)
Ew_target = Ew_target.asfreq('D')
Ew_exog = Ew_exog.asfreq('D')


# In[146]:


Ew_target


# In[147]:


Ew_target = Ew_target.fillna(Ew_target.mean(), inplace = True)  # Forward fill
Ew_exog = Ew_exog.fillna(method='bfill', inplace = True) 


# In[157]:


# ARIMA model with skforecast.Sarimax
# ==============================================================================
warnings.filterwarnings("ignore", category=UserWarning, message='Non-invertible|Non-stationary')
model = Sarimax(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model.fit(y = Ew_target, exog = Ew_exog)
model.summary()
warnings.filterwarnings("default")


# In[159]:


Ew_test = Ew_validX['england_wales_demand']
Ew_exog_test = Ew_validX.drop(columns = ['england_wales_demand'])
# Assign a frequency to the index (e.g., daily data)
Ew_test = Ew_test.asfreq('D')
Ew_exog_test = Ew_exog_test.asfreq('D')


# In[160]:


# Prediction
# ==============================================================================

EwX_predictions = model.predict(steps=len(Ew_test), exog=Ew_exog_test)


EwX_predictions.columns = ['Ew_forecast']
display(EwX_predictions.head())


# In[161]:


# Plot predictions, 
# ==============================================================================
set_dark_theme()

fig, ax = plt.subplots(figsize=(15, 10))
Ew_trainX.england_wales_demand.plot(ax=ax, label='train', color = 'r')
Ew_validX.england_wales_demand.plot(ax=ax, label='test', color = 'g')
EwX_predictions.plot(ax=ax, label='skforecast', color = 'b')
ax.set_title('Predictions with SARIMA models')


# In[162]:


print("MSE of Sarimax for england_wales_demand:", mean_squared_error(Ew_test, EwX_predictions.Ew_forecast))
print("RMSE of Sarimax for england_wales_demand:", np.sqrt(mean_squared_error(Ew_test, EwX_predictions.Ew_forecast)))
print("MAE of Sarimax for england_wales_demand:", mean_absolute_error(Ew_test, EwX_predictions.Ew_forecast))
print("MAPE of Sarimax for england_wales_demand:", mean_absolute_percentage_error(Ew_test, EwX_predictions.Ew_forecast))
print("R2_score of Sarimax for england_wales_demand:", r2_score(Ew_test, EwX_predictions.Ew_forecast))


# ## XGBoost

# In[164]:


pip install xgboost


# In[165]:


import xgboost as xgb


# In[166]:


threshold_date_1 = "01-01-2024"
threshold_date_2 = "05-11-2024"
train_data = sample.loc[sample.index < threshold_date_1]
test_data = sample.loc[(sample.index >= threshold_date_1) & (sample.index < threshold_date_2)]
hold_out_data = sample.loc[sample.index >= threshold_date_2]


# In[167]:


train_data.shape,test_data.shape,hold_out_data.shape


# In[168]:


fig, ax = plt.subplots(figsize=(15, 5))
train_data["National_demand"].plot(ax=ax, label="Training set")
test_data["National_demand"].plot(ax=ax, label="Test set")
hold_out_data["National_demand"].plot(ax=ax, label="Hold-out set")
ax.axvline(threshold_date_1, color="k", ls="--")
ax.axvline(threshold_date_2, color="k", ls=":")
ax.set_title("Training-test split")
plt.legend();


# #### National Demand

# In[169]:


Nd_features = ['Total_demand', 'england_wales_demand',
       'embedded_wind_generation', 'embedded_wind_capacity',
       'embedded_solar_generation', 'embedded_solar_capacity', 'non_bm_stor',
       'pump_storage_pumping', 'ifa_flow', 'ifa2_flow', 'britned_flow',
       'moyle_flow', 'east_west_flow', 'nemo_flow', 'is_holiday']

Nd_target = ['National_demand']


# In[170]:


# Prepare the training, testing and hold-out data
X_train_nd = train_data[Nd_features]
y_train_nd = train_data[Nd_target]

X_test_nd = test_data[Nd_features]
y_test_nd = test_data[Nd_target]

X_hold_out_nd = hold_out_data[Nd_features]
y_hold_out_nd = hold_out_data[Nd_target]

# Initialize and fit the XGBoost model
xgb_simple = xgb.XGBRegressor(
    n_estimators=500, 
    max_depth=3, 
    learning_rate=0.01, 
    early_stopping_rounds=50, 
#     tree_method="gpu_hist",
    random_state=43, 
)

xgb_simple.fit(
    X_train_nd,
    y_train_nd,
    eval_set=[(X_train_nd, y_train_nd), (X_hold_out_nd, y_hold_out_nd)],
    verbose=100,
);


# In[171]:


feat_imp_1 = pd.DataFrame(
    data=xgb_simple.feature_importances_,
    index=xgb_simple.get_booster().feature_names,
    columns=["importance"],
)

feat_imp_1.sort_values("importance", ascending=True, inplace=True)

feat_imp_1.plot(kind="barh");


# In[172]:


# result_frame = y_test_nd.to_frame()
y_test_nd["pred_xgb_Nd"] = xgb_simple.predict(X_test_nd)


# In[173]:


print("MSE of XGBoost for National_demand:", mean_squared_error(y_test_nd["National_demand"], y_test_nd["pred_xgb_Nd"]))
print("RMSE of XGBoost for National_demand:", np.sqrt(mean_squared_error(y_test_nd["National_demand"], y_test_nd["pred_xgb_Nd"])))
print("MAE of XGBoost for National_demand:", mean_absolute_error(y_test_nd["National_demand"], y_test_nd["pred_xgb_Nd"]))
print("MAPE of XGBoost for National_demand:", mean_absolute_percentage_error(y_test_nd["National_demand"], y_test_nd["pred_xgb_Nd"]))
print("R2_score of XGBoost for National_demand:", r2_score(y_test_nd["National_demand"], y_test_nd["pred_xgb_Nd"]))


# In[174]:


fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(y_test_nd.index, y_test_nd["National_demand"], "o", label="Test set")
ax.plot(y_test_nd.index, y_test_nd["pred_xgb_Nd"], "o", label="Prediction")

ax.legend(loc="center", bbox_to_anchor=(1.075, 0.5))

ax.set_title("Prediction on test set")
ax.set_ylabel("Energy Demand (MW)")
ax.set_xlabel("Date");


# In[175]:


begin = "02-10-2024"
end = "02-24-2024"

fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(
    y_test_nd.loc[(y_test_nd.index > begin) & (y_test_nd.index < end)].index,
    y_test_nd.loc[(y_test_nd.index > begin) & (y_test_nd.index < end)]["National_demand"],
    "o",
    label="Test set",
)

ax.plot(
    y_test_nd.loc[(y_test_nd.index > begin) & (y_test_nd.index < end)].index,
    y_test_nd.loc[(y_test_nd.index > begin) & (y_test_nd.index < end)][
        "pred_xgb_Nd"
    ],
    "o",
    label="Prediction",
)

ax.legend(loc="center", bbox_to_anchor=(1.075, 0.5))

ax.set_title("Prediction on test set - Two weeks")
ax.set_ylabel("National Energy Demand (MW)")
ax.set_xlabel("Date");


# In[176]:


plt.figure(figsize = (15, 5))
plt.plot(X_train_nd.index, y_train_nd["National_demand"], label="Training set")
plt.plot(X_test_nd.index, y_test_nd["National_demand"], label="Test set")
plt.plot(X_test_nd.index, y_test_nd["pred_xgb_Nd"], label="Predictions")
plt.axvline(X_test_nd.index.min(), color="y", ls="--")
plt.legend(loc="center", bbox_to_anchor=(1.075, 0.5))

plt.title("Prediction on test set - week")
plt.ylabel("National Energy Demand (MW)")
plt.xlabel("Date")


# #### Transmission Demand

# In[177]:


Td_features = ['National_demand', 'england_wales_demand',
       'embedded_wind_generation', 'embedded_wind_capacity',
       'embedded_solar_generation', 'embedded_solar_capacity', 'non_bm_stor',
       'pump_storage_pumping', 'ifa_flow', 'ifa2_flow', 'britned_flow',
       'moyle_flow', 'east_west_flow', 'nemo_flow', 'is_holiday']

Td_target = ['Total_demand']


# In[178]:


# Prepare the training, testing and hold-out data
X_train_td = train_data[Td_features]
y_train_td = train_data[Td_target]

X_test_td = test_data[Td_features]
y_test_td = test_data[Td_target]

X_hold_out_td = hold_out_data[Td_features]
y_hold_out_td = hold_out_data[Td_target]

xgb_simple.fit(
    X_train_td,
    y_train_td,
    eval_set=[(X_train_td, y_train_td), (X_hold_out_td, y_hold_out_td)],
    verbose=100,
);


# In[179]:


feat_imp_1 = pd.DataFrame(
    data=xgb_simple.feature_importances_,
    index=xgb_simple.get_booster().feature_names,
    columns=["importance"],
)

feat_imp_1.sort_values("importance", ascending=True, inplace=True)

feat_imp_1.plot(kind="barh");


# In[180]:


# result_frame = y_test_nd.to_frame()
y_test_td["pred_xgb_td"] = xgb_simple.predict(X_test_td)


# In[181]:


print("MSE of XGBoost for Transmission_demand:", mean_squared_error(y_test_td["Total_demand"], y_test_td["pred_xgb_td"]))
print("RMSE of XGBoost for Transmission_demand:", np.sqrt(mean_squared_error(y_test_td["Total_demand"], y_test_td["pred_xgb_td"])))
print("MAE of XGBoost for Transmission_demand:", mean_absolute_error(y_test_td["Total_demand"], y_test_td["pred_xgb_td"]))
print("MAPE of XGBoost for Transmission_demand:", mean_absolute_percentage_error(y_test_td["Total_demand"], y_test_td["pred_xgb_td"]))
print("R2_score of XGBoost for Transmission_demand:", r2_score(y_test_td["Total_demand"], y_test_td["pred_xgb_td"]))


# In[182]:


fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(y_test_nd.index, y_test_td["Total_demand"], "o", label="Test set")
ax.plot(y_test_nd.index, y_test_td["pred_xgb_td"], "o", label="Prediction")

ax.legend(loc="center", bbox_to_anchor=(1.075, 0.5))

ax.set_title("Prediction on test set")
ax.set_ylabel("Energy Demand (MW)")
ax.set_xlabel("Date");


# In[183]:


begin = "02-10-2024"
end = "02-24-2024"

fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(
    y_test_td.loc[(y_test_td.index > begin) & (y_test_td.index < end)].index,
    y_test_td.loc[(y_test_td.index > begin) & (y_test_td.index < end)]["Total_demand"],
    "o",
    label="Test set",
)

ax.plot(
    y_test_td.loc[(y_test_td.index > begin) & (y_test_td.index < end)].index,
    y_test_td.loc[(y_test_td.index > begin) & (y_test_td.index < end)][
        "pred_xgb_td"
    ],
    "o",
    label="Prediction",
)

ax.legend(loc="center", bbox_to_anchor=(1.075, 0.5))

ax.set_title("Prediction on test set - Two weeks")
ax.set_ylabel("Transmission Energy Demand (MW)")
ax.set_xlabel("Date");


# In[184]:


plt.figure(figsize = (15, 5))
plt.plot(X_train_td.index, y_train_td["Total_demand"], label="Training set")
plt.plot(X_test_td.index, y_test_td["Total_demand"], label="Test set")
plt.plot(X_test_td.index, y_test_td["pred_xgb_td"], label="Predictions")
plt.axvline(X_test_td.index.min(), color="y", ls="--")
plt.legend(loc="center", bbox_to_anchor=(1.075, 0.5))

plt.title("Prediction on test set - week")
plt.ylabel("Transmission Energy Demand (MW)")
plt.xlabel("Date")


# #### England-Wales Demand

# In[185]:


Ew_features = ['National_demand', 'Total_demand',
       'embedded_wind_generation', 'embedded_wind_capacity',
       'embedded_solar_generation', 'embedded_solar_capacity', 'non_bm_stor',
       'pump_storage_pumping', 'ifa_flow', 'ifa2_flow', 'britned_flow',
       'moyle_flow', 'east_west_flow', 'nemo_flow', 'is_holiday']

Ew_target = ['england_wales_demand']


# In[186]:


# Prepare the training, testing and hold-out data
X_train_EW = train_data[Ew_features]
y_train_EW = train_data[Ew_target]

X_test_EW = test_data[Ew_features]
y_test_EW = test_data[Ew_target]

X_hold_out_EW = hold_out_data[Ew_features]
y_hold_out_EW = hold_out_data[Ew_target]

xgb_simple.fit(
    X_train_EW,
    y_train_EW,
    eval_set=[(X_train_EW, y_train_EW), (X_hold_out_EW, y_hold_out_EW)],
    verbose=100,
);


# In[187]:


feat_imp_1 = pd.DataFrame(
    data=xgb_simple.feature_importances_,
    index=xgb_simple.get_booster().feature_names,
    columns=["importance"],
)

feat_imp_1.sort_values("importance", ascending=True, inplace=True)

feat_imp_1.plot(kind="barh");


# In[188]:


# result_frame = y_test_nd.to_frame()
y_test_EW["pred_xgb_EW"] = xgb_simple.predict(X_test_EW)


# In[189]:


print("MSE of XGBoost for England_wales_demand:", mean_squared_error(y_test_EW["england_wales_demand"], y_test_EW["pred_xgb_EW"]))
print("RMSE of XGBoost for England_wales_demand:", np.sqrt(mean_squared_error(y_test_EW["england_wales_demand"], y_test_EW["pred_xgb_EW"])))
print("MAE of XGBoost for England_wales_demand:", mean_absolute_error(y_test_EW["england_wales_demand"], y_test_EW["pred_xgb_EW"]))
print("MAPE of XGBoost for England_wales_demand:", mean_absolute_percentage_error(y_test_EW["england_wales_demand"], y_test_EW["pred_xgb_EW"]))
print("R2_score of XGBoost for England_wales_demand:", r2_score(y_test_EW["england_wales_demand"], y_test_EW["pred_xgb_EW"]))


# In[190]:


fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(y_test_EW.index, y_test_EW["england_wales_demand"], "o", label="Test set")
ax.plot(y_test_EW.index, y_test_EW["pred_xgb_EW"], "o", label="Prediction")

ax.legend(loc="center", bbox_to_anchor=(1.075, 0.5))

ax.set_title("Prediction on test set")
ax.set_ylabel("English-Welsh Energy Demand (MW)")
ax.set_xlabel("Date");


# In[191]:


begin = "02-10-2024"
end = "02-24-2024"

fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(
    y_test_EW.loc[(y_test_EW.index > begin) & (y_test_EW.index < end)].index,
    y_test_EW.loc[(y_test_EW.index > begin) & (y_test_EW.index < end)]["england_wales_demand"],
    "o",
    label="Test set",
)

ax.plot(
    y_test_EW.loc[(y_test_EW.index > begin) & (y_test_EW.index < end)].index,
    y_test_EW.loc[(y_test_EW.index > begin) & (y_test_EW.index < end)][
        "pred_xgb_EW"
    ],
    "o",
    label="Prediction",
)

ax.legend(loc="center", bbox_to_anchor=(1.075, 0.5))

ax.set_title("Prediction on test set - Two weeks")
ax.set_ylabel("England_Wales Energy Demand (MW)")
ax.set_xlabel("Date");


# In[192]:


plt.figure(figsize = (15, 5))
plt.plot(X_train_EW.index, y_train_EW["england_wales_demand"], label="Training set")
plt.plot(X_test_EW.index, y_test_EW["england_wales_demand"], label="Test set")
plt.plot(X_test_EW.index, y_test_EW["pred_xgb_EW"], label="Prediction")
plt.axvline(X_test_EW.index.min(), color="y", ls="--")
plt.legend(loc="center", bbox_to_anchor=(1.075, 0.5))

plt.title("Prediction on test set - week")
plt.ylabel("England_wales Energy Demand (MW)")
plt.xlabel("Date")


# In[193]:


y_hold_out_EW['Prediction'] = xgb_simple.predict(X_hold_out_EW)
y_hold_out_EW.head()

# X_hold_out_EW
# y_hold_out_EW


# ### RANDOM FOREST

# In[194]:


from sklearn.ensemble import RandomForestRegressor


# #### National Demand

# In[195]:


X_array = X_train_nd.values
y_array = y_train_nd.values.ravel()
# X = array[:,0:-1]
# y = array[:,-1]
model = RandomForestRegressor(n_estimators=500, random_state=1)
model.fit(X_array, y_array)


# In[196]:


RF_pred = model.predict(X_test_nd)
RF_pred[:15]


# In[197]:


ND = y_test_nd['National_demand'].values


# In[198]:


RF_result = pd.DataFrame(ND, RF_pred)
RF_result.reset_index(inplace = True)
RF_result.head()


# In[199]:


RF_result.rename(columns = {'index':'RF_pred_nd', 0:'National_demand'},
                inplace = True)
RF_result.head()


# In[200]:


print("MSE of Random Forest for National_demand:", mean_squared_error(RF_result["National_demand"], RF_result["RF_pred_nd"]))
print("RMSE of Random Forest for National_demand:", np.sqrt(mean_squared_error(RF_result["National_demand"], RF_result["RF_pred_nd"])))
print("MAE of Random Forest for National_demand:", mean_absolute_error(RF_result["National_demand"], RF_result["RF_pred_nd"]))
print("MAPE of Random Forest for National_demand:", mean_absolute_percentage_error(RF_result["National_demand"], RF_result["RF_pred_nd"]))
print("R2_score of Random Forest for National_demand:", r2_score(RF_result["National_demand"], RF_result["RF_pred_nd"]))


# In[201]:


RF_pred_nd_hold_out = model.predict(X_hold_out_nd)


# In[202]:


plt.figure(figsize = (15, 5))
plt.plot(y_train_nd.index, y_train_nd["National_demand"], label="Training set")
plt.plot(y_test_nd.index, RF_result["RF_pred_nd"], label="Prediction")
plt.plot(y_test_nd.index, RF_result["National_demand"], label="Test set")
plt.plot(y_hold_out_nd.index, RF_pred_nd_hold_out, label="Next prediction")
plt.axvline(y_test_nd.index.min(), color="y", ls="--")
plt.legend(loc="center", bbox_to_anchor=(1.075, 0.5))

plt.title("Prediction on test set - week (National Demand)")
plt.ylabel("National Energy Demand (MW)")
plt.xlabel("Date")


# from sklearn.feature_selection import RFE
# rfe = RFE(RandomForestRegressor(n_estimators=500, random_state=1), 4)
# fit = rfe.fit(X, y)
# names = lags.columns
# columns=[]
# for i in range(len(fit.support_)):
#     if fit.support_[i]:
#         columns.append(names[i])
# 
# print("Columns with predictive power:", columns )

# #### Transmission demand

# In[203]:


X_array_td = X_train_td.values
y_array_td = y_train_td.values.ravel()
# X = array[:,0:-1]
# y = array[:,-1]
model = RandomForestRegressor(n_estimators=350, random_state=1)
model.fit(X_array_td, y_array_td)


# In[204]:


RF_pred_td = model.predict(X_test_td)
RF_pred_td[:15]


# In[205]:


TD = y_test_td['Total_demand'].values


# In[206]:


RF_result_td = pd.DataFrame(TD, RF_pred_td)
RF_result_td.reset_index(inplace = True)
RF_result_td.head()


# In[207]:


RF_result_td.rename(columns = {'index':'RF_pred_td', 0:'Total_demand'},
                inplace = True)
RF_result_td.head()


# In[208]:


print("MSE of Random Forest for Total_demand:", mean_squared_error(RF_result_td["Total_demand"], RF_result_td["RF_pred_td"]))
print("RMSE of Random Forest for Total_demand:", np.sqrt(mean_squared_error(RF_result_td["Total_demand"], RF_result_td["RF_pred_td"])))
print("MAE of Random Forest for Total_demand:", mean_absolute_error(RF_result_td["Total_demand"], RF_result_td["RF_pred_td"]))
print("MAPE of Random Forest for Total_demand:", mean_absolute_percentage_error(RF_result_td["Total_demand"], RF_result_td["RF_pred_td"]))
print("R2_score of Random Forest for Total_demand:", r2_score(RF_result_td["Total_demand"], RF_result_td["RF_pred_td"]))


# In[209]:


RF_pred_td_hold_out = model.predict(X_hold_out_td)


# In[210]:


plt.figure(figsize = (15, 5))
plt.plot(y_train_td.index, y_train_td["Total_demand"], label="Training set")
plt.plot(y_test_td.index, RF_result_td["RF_pred_td"], label="Prediction")
plt.plot(y_test_td.index, RF_result_td["Total_demand"], label="Test set")
plt.plot(y_hold_out_td.index, RF_pred_td_hold_out, label="Next prediction")
plt.axvline(y_test_nd.index.min(), color="y", ls="--")
plt.legend(loc="center", bbox_to_anchor=(1.075, 0.5))

plt.title("Prediction on test set - week (Total Demand)")
plt.ylabel("Total Energy Demand (MW)")
plt.xlabel("Date")


# #### English-Welsh Energy demand

# In[211]:


X_array_EW = X_train_EW.values
y_array_EW = y_train_EW.values.ravel()
# X = array[:,0:-1]
# y = array[:,-1]
model = RandomForestRegressor(n_estimators=350, random_state=1)
model.fit(X_array_EW, y_array_EW)


# In[215]:


RF_pred_EW = model.predict(X_test_EW)
RF_pred_EW[:15]


# In[216]:


EW = y_test_EW['england_wales_demand'].values


# In[217]:


RF_result_EW = pd.DataFrame(EW, RF_pred_EW)
RF_result_EW.reset_index(inplace = True)
RF_result_EW.head()


# In[218]:


RF_result_EW.rename(columns = {'index':'RF_pred_EW', 0:'england_wales_demand'},
                inplace = True)
RF_result_EW.head()


# In[219]:


print("MSE of Random Forest for england_wales_demand:", mean_squared_error(RF_result_EW["england_wales_demand"], RF_result_EW["RF_pred_EW"]))
print("RMSE of Random Forest for england_wales_demand:", np.sqrt(mean_squared_error(RF_result_EW["england_wales_demand"], RF_result_EW["RF_pred_EW"])))
print("MAE of Random Forest for england_wales_demand:", mean_absolute_error(RF_result_EW["england_wales_demand"], RF_result_EW["RF_pred_EW"]))
print("MAPE of Random Forest for england_wales_demand:", mean_absolute_percentage_error(RF_result_EW["england_wales_demand"], RF_result_EW["RF_pred_EW"]))
print("R2_score of Random Forest for england_wales_demand:", r2_score(RF_result_EW["england_wales_demand"], RF_result_EW["RF_pred_EW"]))


# In[222]:


RF_pred_EW_hold_out = model.predict(X_hold_out_EW)


# In[223]:


plt.figure(figsize = (15, 5))
plt.plot(y_train_EW.index, y_train_EW["england_wales_demand"], label="Training set")
plt.plot(y_test_EW.index, RF_result_EW["RF_pred_EW"], label="Prediction")
plt.plot(y_test_EW.index, RF_result_EW["england_wales_demand"], label="Test set")
plt.plot(y_hold_out_EW.index, RF_pred_EW_hold_out, label="Next prediction")
plt.axvline(y_test_nd.index.min(), color="y", ls="--")
plt.legend(loc="center", bbox_to_anchor=(1.075, 0.5))

plt.title("Prediction on test set - week (england_wales_demand)")
plt.ylabel("England-Wales Energy Demand (MW)")
plt.xlabel("Date")

