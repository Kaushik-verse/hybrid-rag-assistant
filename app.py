import streamlit as st
from rag import (
    process_inputs, generate_answer, clear_database,
    calculate_roi, format_roi_report, format_indian_currency
)

st.set_page_config(
    page_title="AUREA | Real Estate AI",
    page_icon="🏡",
    layout="wide",
)

st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f7;
    }
    .stChatInputContainer {
        border-radius: 20px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        background-color: rgba(255, 255, 255, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .stSidebar {
        background-color: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("🏗️ Agent Settings")

    model_choice = st.selectbox(
        "AI Analyst Model",
        ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        index=0,
    )

    answer_style = st.selectbox(
        "Analysis Persona",
        ["Investor", "Homebuyer", "Legal Expert"],
        index=0,
    )

    st.divider()
    st.subheader("📁 Property Data Source")

    url1 = st.text_input("Listing URL 1")
    url2 = st.text_input("Listing URL 2")

    uploaded_pdfs = st.file_uploader(
        "Upload Contracts / Brochures",
        type="pdf",
        accept_multiple_files=True,
    )

    if st.button("Analyze Property Data 🔍", type="primary"):
        urls = [u for u in [url1, url2] if u.strip()]
        if not urls and not uploaded_pdfs:
            st.error("Please provide a URL or PDF.")
        else:
            with st.status("Processing property data...", expanded=True):
                for msg in process_inputs(urls, uploaded_pdfs):
                    st.write(msg)

    st.divider()

    with st.expander("💰 Quick Mortgage Calculator"):
        price = st.number_input("Home Price (₹)", value=5000000, step=100000)
        down = st.number_input("Down Payment (₹)", value=1000000, step=50000)
        rate = st.slider("Interest Rate (%)", 1.0, 12.0, 8.5, 0.1)
        years = st.selectbox("Loan Term", [15, 20, 30], index=1)
        loan = price - down
        r = rate / 100 / 12
        n = years * 12
        payment = loan * (r * (1 + r) ** n) / ((1 + r) ** n - 1) if r else loan / n
        st.metric("Monthly Payment", format_indian_currency(payment))

    st.divider()

    chat_log = ["REAL ESTATE ANALYSIS REPORT\n" + "=" * 40]
    for m in st.session_state.messages:
        chat_log.append(f"\n[{m['role'].upper()}]\n{m['content']}")
        if "sources" in m:
            chat_log.append(f"\nSOURCES:\n{m['sources']}")

    st.download_button(
        "📥 Download Report",
        "\n".join(chat_log),
        "property_analysis_report.txt",
        "text/plain",
    )

    if st.button("Clear Current Property 🗑️"):
        st.session_state.messages = []
        st.success(clear_database())

tab_chat, tab_roi = st.tabs(["💬 AI Chat", "📊 ROI Calculator"])

with tab_chat:
    st.title("🏡 Real Estate AI Agent")
    st.caption(
        f"Persona: **{answer_style}** | Model: **{model_choice}** | "
        "Retrieval: **Hybrid BM25 + Dense + Rerank**"
    )

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if "sources" in m and m["sources"]:
                with st.expander("📊 Sources"):
                    st.markdown(m["sources"])

    if prompt := st.chat_input("Ask about price, ROI, zoning, valuation…"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing with hybrid search…"):
                answer, docs = generate_answer(prompt, model_choice, answer_style)
                st.markdown(answer)

                sources = ""
                for i, d in enumerate(docs):
                    src = d.metadata.get("source", "Unknown")
                    preview = d.page_content[:200].replace("\n", " ") + "…"
                    sources += f"**Source {i + 1} ({src})**\n>{preview}\n\n"

                if sources:
                    with st.expander("📊 Sources"):
                        st.markdown(sources)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

with tab_roi:
    st.header("📊 Investment ROI Calculator")
    st.markdown(
        "Wondering if a property is a good investment? Fill out the basics below, and we'll calculate your potential profits and cash flow.")

    st.subheader("Step 1: The Basics")
    col1, col2 = st.columns(2)
    with col1:
        r_price = st.number_input("Purchase Price (₹)", value=40000000, step=500000, key="roi_price",
                                  help="The total cost to buy the property.")
        r_rent = st.number_input("Expected Monthly Rent (₹)", value=120000, step=5000, key="roi_rent",
                                 help="How much rent you expect to collect every month.")

    with col2:
        r_down_pct = st.slider("Down Payment (%)", 5, 50, 20, key="roi_down",
                               help="The percentage of the price you are paying upfront in cash.")
        r_rate = st.slider("Home Loan Interest Rate (%)", 5.0, 12.0, 8.5, 0.1, key="roi_rate")

    with st.expander("⚙️ Step 2: Advanced Expenses & Assumptions (Optional)"):
        st.markdown("We've filled these in with standard market averages, but feel free to adjust them.")
        ac1, ac2 = st.columns(2)
        with ac1:
            r_term = st.selectbox("Loan Term (years)", [10, 15, 20, 30], index=2, key="roi_term")
            r_appr = st.slider("Annual Property Value Growth (%)", 0.0, 15.0, 5.0, 0.5, key="roi_appr",
                               help="How much you expect the property's value to increase each year.")
            r_vacancy = st.slider("Vacancy Rate (%)", 0.0, 20.0, 5.0, 0.5, key="roi_vac",
                                  help="Accounts for months when the property sits empty between tenants.")
        with ac2:
            r_tax = st.number_input("Annual Property Tax (₹)", value=int(r_price * 0.01), step=5000, key="roi_tax")
            r_insure = st.number_input("Annual Insurance (₹)", value=15000, step=1000, key="roi_ins")
            r_maint = st.number_input("Annual Maintenance (₹)", value=int(r_price * 0.005), step=5000, key="roi_maint")

    st.divider()

    if st.button("Calculate My ROI 🚀", type="primary", key="roi_btn"):
        total_exp = r_tax + r_insure + r_maint
        result = calculate_roi(
            purchase_price=r_price,
            down_payment_pct=r_down_pct,
            interest_rate=r_rate,
            loan_term_years=r_term,
            monthly_rental=r_rent,
            vacancy_rate=r_vacancy,
            annual_expenses=total_exp,
            appreciation_rate=r_appr,
        )

        st.subheader("Your Investment at a Glance")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Monthly Cash Flow", f"{format_indian_currency(result.monthly_cash_flow)}",
                  delta="Profitable" if result.monthly_cash_flow > 0 else "Losing Money",
                  help="The actual cash left in your pocket each month after paying the mortgage and expenses.")

        k2.metric("Cash-on-Cash Return", f"{result.cash_on_cash_return:.1f}%",
                  help="If you put this cash in the bank, what interest rate would it earn? This is the real estate equivalent.")

        k3.metric("Cap Rate", f"{result.cap_rate:.1f}%",
                  help="The annual return on the property if you bought it entirely in cash (no loan).")

        k4.metric("5-Year Total ROI", f"{result.total_roi_5yr:.0f}%",
                  help="Your total estimated profit in 5 years, including rent, paying down the loan, and the property's value going up.")

        st.markdown(format_roi_report(result))