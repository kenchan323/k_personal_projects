#streamlit run "C:\dev\k_personal_projects\others\streamlit_fun\sample.py"
import pandas as pd
import streamlit as st
from utils.data.source import YahooFinanceDB
import datetime

import time

def load_yf_db_timeseries(ticker, fld='Close', start=pd.Timestamp(2025, 1, 1),
                          end=pd.Timestamp(2025, 6, 1)):
    return yf_db.load_timeseries(ids=ticker,
                                     fld=fld,
                                     start=start,
                                     end=end)



def plot_ticker_px(ticker, start_date, end_date, to_rebase=False):
    #st.markdown(f"Ticker is {ticker}")
    # NASDAQ Composite Index
    if not ticker:
        st.warning("Please pick at least one security")
        return
    st.markdown(ticker)
    bmk_p_ts = load_yf_db_timeseries(ticker, fld='Close',
                                     start=start_date,
                                     end=end_date).copy(deep=True)
    if bmk_p_ts.empty:
        st.error(f"{ticker} did not return any data")
    if to_rebase:
        # backfill missing data at the start with the first valid in each column
        for _k, _first in zip(bmk_p_ts.keys(), [bmk_p_ts[k].first_valid_index() for k in bmk_p_ts]):
            bmk_p_ts.loc[:_first] = bmk_p_ts.loc[:_first].bfill()
        bmk_p_ts /= bmk_p_ts.iloc[0]
    st.line_chart(bmk_p_ts, use_container_width=True)

yf_db = YahooFinanceDB()

NAME_TO_TICKER = {'NVDIA': 'NVDA', 'VIX': '^VIX', 'FTSE 100': '^FTSE', 'MOVE': '^MOVE', 'NASDAQ': '^IXIC',
                  'EMPTY': 'EMPTY'}
chosen_instrument = st.multiselect('Pick a security', list(NAME_TO_TICKER.keys()))
start_date = st.date_input("From", datetime.date(2000, 1, 1))
end_date = st.date_input("To", None)
to_rebase = st.radio('Rebase series?', [True, False])

def my_callback(text_to_display):
    st.text(str(text_to_display))


st.button("plot_px", on_click=plot_ticker_px, args=([NAME_TO_TICKER[x] for x in chosen_instrument],
                                                    start_date, end_date,
                                                    to_rebase))
@st.cache
def return_same(a):
    time.sleep(5)
    return a

def markdown_a(a):
    st.markdown(return_same(a))
st.button("return!", on_click=markdown_a, args=(3,))

st.markdown(2)

st.title("This is the app title")
st.header("This is the header")
st.markdown("This is the markdown")
st.subheader("This is the subheader")
st.caption("This is the caption")
st.code("x = 2025")
st.latex(r''' a+a r^1+a r^2+a r^3 ''')


st.image("data:image/jpeg;base64,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", caption="A kid playing")
# st.audio("audio.mp3")
# st.video("video.mp4")

st.checkbox('Yes')
st.button('Click Me')
st.radio('Pick your gender', ['Male', 'Female'])
st.selectbox('Pick a fruit', ['Apple', 'Banana', 'Orange'])
st.multiselect('Choose a planet', ['Jupiter', 'Mars', 'Neptune'])
st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])
st.slider('Pick a number', 0, 50)
st.progress(10)

st.success("You did it!")
st.error("Error occurred")
st.warning("This is a warning")
st.info("It's easy to build a Streamlit app")
st.exception(RuntimeError("RuntimeError exception"))


st.sidebar.title("Sidebar Title")
st.sidebar.markdown("This is the sidebar content")

with st.container():
    st.write("This is inside the container")
    st.success("You did it! inside the container")




@st.cache
def get_fvalue(val):
    feature_dict = {"No": 1, "Yes": 2}
    return feature_dict[val]
# def get_value(val, my_dict):
#     return my_dict[val]
#
# if app_mode == 'Home':
#     st.title('Loan Prediction')
#     st.image('loan_image.jpg')
#     st.markdown('Dataset:')
#     data = pd.read_csv('loan_dataset.csv')
#     st.write(data.head())
#     st.bar_chart(data[['ApplicantIncome', 'LoanAmount']].head(20)

