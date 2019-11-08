import pandas as pd

class DataLoader:
    def __init__(self, filepath, separator, rowFrom, rowTo, columnFrom, columnTo, all):
        df = pd.read_csv(filepath, sep=separator)
        if all == True:
            self.dataframe = df
        else:
            self.dataframe = df.iloc[rowFrom:rowTo, columnFrom:columnTo]

        # self.selectedColumns = ['Happiness.Score', 'Freedom', 'Family',  'Population age distribution <14 (in %)',
        #     'Population age distribution >60 (in %)', 'Population growth rate (average annual %)',
        #     'Economy.GDP.per.Capita.', 'GDP: Gross domestic product (million current US$)',
        #     'GDP per capita (current US$)', 'GDP growth rate (annual % const. 2005 prices)',
        #     'Life expectancy at birth - females', 'Life expectancy at birth - males',
        #     'Infant mortality rate (per 1000 live births)', 'Health: Total expenditure (% of GDP)',
        #      'Energy supply per capita (Gigajoules)',
        #     'Fertility rate total (live births per woman)', 'Individuals using the Internet (per 100 inhabitants)']
    def printData(self):
        print(self.dataframe)

    def printDataAsArray(self):
        print(self.dataframe.values)

    def printDataInfo(self):
        print(self.dataframe.info())

    def printWideData(self):
        with pd.option_context('display.max_rows', None, 'display.max_columns',None):
            print(self.dataframe)