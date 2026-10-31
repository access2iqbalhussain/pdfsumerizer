import gradio as gr
import torch
from pypdf import PdfReader
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)

SIZES = {"Short": (60, 25), "Medium": (130, 50), "Detailed": (200, 90)}


def read_pdf(path):
    reader = PdfReader(path)
    return "\n".join((p.extract_text() or "") for p in reader.pages).strip()


def chunk_text(text, words_per_chunk=500):
    words = text.split()
    return [" ".join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]


def summarize_chunk(chunk, max_len, min_len):
    n = len(chunk.split())
    max_len = min(max_len, max(20, int(n * 0.8)))
    min_len = max(5, min(min_len, max_len - 5))
    out = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False, truncation=True)
    return out[0]["summary_text"]


def summarize(text, size):
    max_len, min_len = SIZES[size]
    chunks = chunk_text(text)
    partial = [summarize_chunk(c, max_len, min_len) for c in chunks]
    combined = " ".join(partial)
    if len(chunks) > 1:
        if len(combined.split()) > 500:
            return summarize(combined, size)
        return summarize_chunk(combined, max_len + 40, min_len)
    return combined


def from_pdf(file, size):
    if file is None:
        return "Please upload a PDF first."
    text = read_pdf(file)
    if not text:
        return "No text found in this PDF (it may be scanned images)."
    return summarize(text, size)


def from_text(text, size):
    if not text or not text.strip():
        return "Please paste some text first."
    return summarize(text.strip(), size)


with gr.Blocks(title="AI Summarizer") as demo:
    gr.Markdown("# AI Summarizer\nSummarize a PDF or long text. English only.")
    size = gr.Radio(list(SIZES), value="Medium", label="Summary length")
    with gr.Tabs():
        with gr.Tab("Upload PDF"):
            pdf = gr.File(label="PDF file", file_types=[".pdf"], type="filepath")
            b1 = gr.Button("Summarize PDF", variant="primary")
            o1 = gr.Textbox(label="Summary", lines=10, show_copy_button=True)
        with gr.Tab("Paste long text"):
            txt = gr.Textbox(label="Long text", lines=12, placeholder="Paste your text here...")
            b2 = gr.Button("Summarize text", variant="primary")
            o2 = gr.Textbox(label="Summary", lines=10, show_copy_button=True)
    b1.click(from_pdf, [pdf, size], o1)
    b2.click(from_text, [txt, size], o2)

demo.queue().launch()
