import streamlit as st
import duckdb
import pandas as pd
from datetime import date
import yfinance as yf
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib
from talib import MA_Type

# Example DataFrame


# Скачивание данных о котировках
def download_stock_data(ticker, start_date: date, end_date: date):
    data = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), interval="1d", threads=True, auto_adjust=True, group_by='Ticker')
    return data

def preload_date_for_ticker(db_connection, ticker: str, today: date) -> pd.DataFrame:
    start_date = today
    last_item = db_connection.sql("select stock_date from stock_data where ticker like '{}' order by stock_date desc limit 1".format(ticker))   
    if last_item.df().shape[0] == 0:
        start_date = today - relativedelta(years=1)
    else:
        start_date = last_item.df().iloc[0]['stock_date']
    
    print(start_date, "->", ticker)
    
    data = download_stock_data(ticker, start_date, today)[ticker]
    # Заполняем пропуски
    data.ffill(inplace=True)
    # Сглаживаем выбросы
    get_rid_of_outliers(data)

    for index, row in data.iterrows():        
        insert_statement = """
            insert into stock_data(ticker, stock_date, open, close, high, low, volume, is_outlier) values('{}', '{}', {}, {}, {}, {}, {}, {})  ON CONFLICT DO NOTHING;
        """.format(
                   ticker,
                   index,
                   row['Open'],
                   row['Close'],
                   row['High'],
                   row['Low'],
                   row['Volume'],
                   row['is_outlier'])
        print(insert_statement)
        db_connection.sql(insert_statement)

def establish_db_connection():
    con = duckdb.connect("./stock_data.db")

    create_table_statement = """
        CREATE TABLE IF NOT EXISTS stock_data (
            ticker varchar(10) not null,
            stock_date datetime not null,
            open FLOAT not null,
            close FLOAT not null,
            high FLOAT not null,
            low FLOAT not null,
            volume FLOAT not null,
            is_outlier boolean not null
        );
        
    """

    create_index_statement = """
        CREATE UNIQUE INDEX ticker_ts ON stock_data (ticker, stock_date);
    """

    con.sql(create_table_statement)

    try:
        con.sql(create_index_statement)
    except:
        print("index alredy exists")

    return con


def get_rid_of_outliers(source_df: pd.DataFrame): 
    print(source_df.columns)
    # Вычисление межквартильного размаха (IQR)
    Q1 = source_df['Close'].quantile(0.25)  # Первый квартиль (25-й процентиль)
    Q3 = source_df['Close'].quantile(0.75)  # Третий квартиль (75-й процентиль)
    IQR = Q3 - Q1                        # Межквартильный размах

    # Определение границ для выбросов
    lower_bound = Q1 - 1.5 * IQR  # Нижняя граница
    upper_bound = Q3 + 1.5 * IQR  # Верхняя граница

    # Пометка выбросов
    source_df['is_outlier'] = (source_df['Close'] < lower_bound) | (source_df['Close'] > upper_bound)
    window = 5  # Количество дней для среднего
    source_df['price_corrected'] = source_df['Close']
    source_df.loc[source_df['is_outlier'], 'Close'] =  source_df['price_corrected'].rolling(window=window, center=True).mean()
    source_df.drop(columns=['price_corrected'], inplace=True)
    source_df.ffill(inplace=True)
   

def add_features(source_df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    for ticker in source_df['ticker'].unique():
            ticker_df = source_df[source_df['ticker'] == ticker]
            ticker_df['scaled_close_price'] = scaler.fit_transform(ticker_df[['close']])

            ticker_df['tema'] = talib.TEMA(ticker_df['close'], timeperiod=24)
            ticker_df['macd'], ticker_df['macd_signal_line'], ticker_df['macd_hist'] = talib.MACD(ticker_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            
            # Создаем сигналы для покупки и продажи
            ticker_df['macd_signal'] = 0
            ticker_df.loc[(ticker_df['macd'] > ticker_df['macd_signal_line']) & (ticker_df['close'] > ticker_df['tema']), 'macd_signal'] = 1  # Сигнал на покупку
            ticker_df.loc[(ticker_df['macd'] < ticker_df['macd_signal_line']) & (ticker_df['close'] < ticker_df['tema']), 'macd_signal'] = -1  # Сигнал на продажу


            source_df.loc[source_df['ticker'] == ticker, 'scaled_close_price'] = ticker_df['scaled_close_price']
            source_df.loc[source_df['ticker'] == ticker, 'macd'] = ticker_df['macd']
            source_df.loc[source_df['ticker'] == ticker, 'macd_signal_line'] = ticker_df['macd_signal_line']
            source_df.loc[source_df['ticker'] == ticker, 'macd_signal'] = ticker_df['macd_signal']
            source_df.loc[source_df['ticker'] == ticker, 'macd'] = ticker_df['macd']
            source_df.loc[source_df['ticker'] == ticker, 'tema'] = ticker_df['tema']

    source_df['daily_return'] = source_df.groupby('ticker')['close'].pct_change()
    return source_df[1:]

def update_dataframe_ui(con, ui_loaded): 
    df = con.sql("select * from stock_data;").df()
    if df.shape[0] != 0 and not ui_loaded:
        ui_loaded = True
        selected_ticker = st.sidebar.multiselect("Select Ticker", options=df['ticker'].unique(), default=df['ticker'].unique())
        start_date =  df['stock_date'].max() - relativedelta(years=1)
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(df['stock_date'].min(), df['stock_date'].max()),
            min_value=df['stock_date'].min(),
            max_value=df['stock_date'].max()
        )

        # Convert date_range to datetime (if tuple provided)
        if isinstance(date_range, tuple):
            start_date, end_date = date_range
        else:
            start_date, end_date = df['Date'].min(), df['Date'].max()

        # Main Dashboard
        st.title("Loaded Assets")
        filtered_df = df[(df['stock_date'] >= pd.to_datetime(start_date)) & (df['stock_date'] <= pd.to_datetime(end_date)) & (df['ticker'].isin(selected_ticker))]
       
        filtered_df = add_features(filtered_df)

        main_table = st.dataframe(filtered_df, key="main_table")

                # Create the figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,  # Share x-axis for easier comparison
            vertical_spacing=0.1,  # Space between plots
            subplot_titles=("Price Over Time", "Daily Returns")  # Titles for subplots
        )

        # Add price traces for each ticker
        for ticker in filtered_df['ticker'].unique():
            ticker_df = filtered_df[df['ticker'] == ticker]
            fig.add_trace(
                go.Scatter(
                    x=ticker_df['stock_date'],
                    y=ticker_df['scaled_close_price'],
                    mode='lines',
                    name=f'{ticker} Price'
                ),
                row=1, col=1
            )

        # Add daily return traces for each ticker
        for ticker in filtered_df['ticker'].unique():
            ticker_df = filtered_df[filtered_df['ticker'] == ticker]
            fig.add_trace(
                go.Scatter(
                    x=ticker_df['stock_date'],
                    y=ticker_df['daily_return'],
                    mode='lines',
                    name=f'{ticker} Returns'
                ),
                row=2, col=1
            )

        # Update layout for better visualization
        fig.update_layout(
            height=700,  # Adjust height
            title_text="Price and Daily Returns by Ticker",
            xaxis_title="Date",
            yaxis_title="Price",
            yaxis2_title="Daily Returns",
            template='plotly'
        )

        st.plotly_chart(fig, use_container_width=False, on_select="ignore", selection_mode=('points', 'box', 'lasso'))
        return True
    return False

def main():
    st.sidebar.header("Filters")
    con = establish_db_connection()
    ui_loaded = update_dataframe_ui(con, False)
    
    if st.sidebar.button("Update stock prices"):
        # Event triggered when button is clicked
        tickers_to_download = ['AAPL', 'GOOG','AMZN', 'MSFT', 'AMD', 'NVDA', 'IBM']
        success_element = None
        for t in tickers_to_download:
            preload_date_for_ticker(con, t,  date.today())
            if success_element:
                 success_element.empty()
            success_element = st.success("{} prices were updated".format(t))
        if success_element:
            success_element.empty()           
        ui_loaded = update_dataframe_ui(con, ui_loaded)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Load asset demo", page_icon=":chart_with_upwards_trend:"
    )
    main()

