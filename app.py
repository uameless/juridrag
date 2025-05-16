import streamlit as st

st.set_page_config(page_title="JuridRAG", layout="wide")

st.markdown("""
    <style>
    html, body, [class*="css"] {
        direction: rtl;
        text-align: right;
        font-family: "Arial", sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

from utils import load_pdf, split_docs, load_arabic_embeddings, create_vector_store
from logic import generate_summary, setup_qa_chain

st.title("📚 JuridRAG 📚")

uploaded_file = st.file_uploader("📤 حمّل وثيقة قانونية (PDF)", type=["pdf"])

if uploaded_file:
    st.success(f"✅ تم تحميل الملف: {uploaded_file.name}")
    st.write("📄 الحجم:", round(uploaded_file.size / 1024, 2), "KB")

    with st.spinner("📄 جارٍ معالجة الوثيقة..."):
        docs = load_pdf(uploaded_file)

        if not docs:
            st.error("❌ لم يتم استخراج أي نص من الوثيقة. تأكد من أن الملف يحتوي على نصوص قابلة للقراءة.")
            st.stop()

        full_text = "\n".join([doc.page_content for doc in docs])
        chunks = split_docs(docs)

        if not chunks:
            st.error("❌ لم يتم تقسيم النص إلى مقاطع. تحقق من محتوى الوثيقة.")
            st.stop()

        embeddings = load_arabic_embeddings()
        vectorstore = create_vector_store(chunks, embeddings)
        qa_chain = setup_qa_chain(vectorstore)

    tab1, tab2 = st.tabs(["📝 توليد الملخص", "💬 محادثة قانونية"])

    with tab1:
        if st.button("🔍 توليد الملخص"):
            with st.spinner("✍️ يتم تلخيص الوثيقة..."):
                summary = generate_summary(full_text)
                st.markdown("📑 **الملخص:**", unsafe_allow_html=True)
                st.markdown(
                    f"""<div dir="rtl" style="text-align: right; font-size: 18px; line-height: 1.8;">{summary}</div>""",
                    unsafe_allow_html=True
                )

    with tab2:
        st.subheader("💬 اطرح سؤالك حول الوثيقة")
        question = st.text_input("❓ سؤالك:")

        if question:
            with st.spinner("🔎 جارٍ البحث عن الجواب..."):
                result = qa_chain.invoke({"query": question})
                answer = result["result"]
                st.markdown(
                    f"""<div dir="rtl" style="text-align: right; font-size: 18px; line-height: 1.8;">✅ <b>الجواب:</b><br>{answer}</div>""",
                    unsafe_allow_html=True
                )
