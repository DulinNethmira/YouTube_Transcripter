from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from transcript_models import TranscriptRequest, YouTubeAnalysisResponse, QuizResponse

load_dotenv()

summarizer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
response_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").with_structured_output(YouTubeAnalysisResponse)
quiz_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").with_structured_output(QuizResponse)

app = FastAPI(title="AI Video Analysis Tool")

def video_transcript_analyzer(request: TranscriptRequest) -> YouTubeAnalysisResponse:
    loader = YoutubeLoader.from_youtube_url(request.url, add_video_info=False)
    docs = loader.load()

    transcript_text = "\n".join([doc.page_content for doc in docs])

    summarizer_prompt = ChatPromptTemplate.from_template(
        "Extract video title, description, and uploaded date:\n\n{docs}"
    )
    summarizer_chain = summarizer_prompt | summarizer_llm
    transcript_summary = summarizer_chain.invoke({"docs": transcript_text})

    response_prompt = ChatPromptTemplate.from_template(
        """
        You are an expert YouTube video transcripter.

        Here is the summarized transcript for the video: {video_url}

        {transcript_summary}

        Provide:
        1. Title
        2. Key topics
        3. Recommended audience
        4. Summary
        """
    )

    response_chain = response_prompt | response_llm
    response = response_chain.invoke({
        "video_url": request.url,
        "transcript_summary": transcript_summary
    })

    return response

def video_quiz_generator(request: TranscriptRequest) -> QuizResponse:
    loader = YoutubeLoader.from_youtube_url(request.url, add_video_info=False)
    docs = loader.load()

    transcript_text = "\n".join([doc.page_content for doc in docs])

    summarizer_prompt = ChatPromptTemplate.from_template(
        "Summarize this transcript:\n{docs}"
    )
    summarizer_chain = summarizer_prompt | summarizer_llm
    transcript_summary = summarizer_chain.invoke({"docs": transcript_text})

    quiz_prompt = ChatPromptTemplate.from_template(
        """
        You are a professional educational quiz generator.

        Based on this summary:

        {transcript_summary}

        Create 10 MCQs with:
        - Question
        - Four options (A, B, C, D)
        - Correct answer
        """
    )

    quiz_chain = quiz_prompt | quiz_llm
    quiz = quiz_chain.invoke({"transcript_summary": transcript_summary})

    return quiz

@app.post("/video-transcripter", response_model=YouTubeAnalysisResponse)
def video_transcript_analyzer_endpoint(request: TranscriptRequest):
    return video_transcript_analyzer(request)

@app.post("/video-quiz", response_model=QuizResponse)
def quiz_endpoint(request: TranscriptRequest):
    return video_quiz_generator(request)
