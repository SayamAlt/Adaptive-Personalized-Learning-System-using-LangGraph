from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os, warnings, json, requests, re, ast, arxiv
warnings.filterwarnings("ignore")
from langchain_openai import ChatOpenAI
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import YouTubeSearchTool
from langchain_core.caches import InMemoryCache
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from pathlib import Path
from langchain.tools import tool
from typing import Optional, Literal, List, Dict
from datetime import datetime

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

# Initialize cache
cache = InMemoryCache()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, api_key=OPENAI_API_KEY, cache=cache, max_tokens=4000)

# Tavily Search Tool
def get_tavily_tool():
    key = os.getenv("TAVILY_API_KEY")
    if not key: return None
    try:
        return TavilySearchResults(api_key=key, max_results=2)
    except Exception:
        return None

# Google Scholar Search Tool
def get_google_scholar_tool():
    key = os.getenv("SERPAPI_API_KEY") or os.getenv("SERP_API_KEY")
    if not key: return None
    try:
        from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
        return GoogleScholarQueryRun(api_wrapper=GoogleScholarAPIWrapper(serp_api_key=key))
    except (ImportError, Exception):
        return None

tavily_search_tool = get_tavily_tool()
google_scholar_search_tool = get_google_scholar_tool()
youtube_search_tool = YouTubeSearchTool() 

def download_image(url: str, filename_prefix: str) -> Optional[str]:
    """
    Download an image and save it locally in the 'images/' directory.
    Returns the local relative path or None if download fails.
    """
    if not url or not url.startswith("http"):
        return None
        
    try:
        images_dir = Path("images")
        images_dir.mkdir(exist_ok=True)
        
        # Determine file extension
        ext = url.split('.')[-1].split('?')[0]
        if ext.lower() not in ['jpg', 'jpeg', 'png', 'svg', 'webp']:
            ext = 'png'
            
        local_filename = f"{filename_prefix}_{hash(url) % 100000}.{ext}"
        local_path = images_dir / local_filename
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
            return str(local_path)
    except Exception as e:
        print(f"Error downloading image: {e}")
    return None

def process_lesson_images(content: str) -> str:
    """
    Find Markdown image links in content, download images, and replace with local paths.
    """
    pattern = r"!\[(.*?)\]\((http[s]?://.*?)\)"
    
    def replace_match(match):
        alt_text = match.group(1)
        url = match.group(2)
        local_path = download_image(url, "lesson")
        if local_path:
            return f"![{alt_text}]({local_path})"
        return match.group(0)
        
    return re.sub(pattern, replace_match, content)

def generate_openai_image(prompt: str, context: str = "") -> Optional[str]:
    """
    Generate an image using OpenAI's Image Generation API with balanced cost/quality.
    Uses gpt-image-1 with medium quality for sharp, clear educational illustrations.
    
    Args:
        prompt: The image generation prompt
        context: Brief context/topic for meaningful filename (e.g., "transformer_architecture")
    """
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found. Skipping image generation.")
        return None
    
    print(f"DEBUG [OpenAI]: Generating image with gpt-image-1...")
    
    try:
        from openai import OpenAI
        import base64
        import re
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Balanced settings: gpt-image-1, medium quality for sharp images at 4x baseline cost
        # Enhanced prompt emphasizing accurate spelling and professional text rendering
        enhanced_prompt = (
            f"{prompt}. "
            "CRITICAL: All text labels must be spelled CORRECTLY with perfect grammar and spelling. "
            "Use clear, legible fonts. Double-check all text for accuracy. "
            "Educational diagram style, professional quality, accurate technical terminology."
        )
        
        response = client.images.generate(
            model="gpt-image-1",
            prompt=enhanced_prompt,
            size="1024x1024",
            quality="medium",
            n=1
        )
        
        print(f"DEBUG [OpenAI]: Image generated successfully")
        
        # Check response format
        image_data = response.data[0]
        
        images_dir = Path("images")
        images_dir.mkdir(exist_ok=True)
        
        # Generate meaningful filename from context
        if context:
            # Clean context for filename (remove special chars, lowercase, replace spaces with underscores)
            clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', context.lower())
            clean_name = re.sub(r'\s+', '_', clean_name.strip())
            filename = f"{clean_name[:50]}.png"  # Limit length
        else:
            filename = f"educational_illustration_{hash(prompt) % 10000}.png"
        
        local_path = images_dir / filename
        
        # Handle base64 response (when URL is None)
        if hasattr(image_data, 'b64_json') and image_data.b64_json:
            print(f"DEBUG [OpenAI]: Processing base64 image data...")
            image_bytes = base64.b64decode(image_data.b64_json)
            with open(local_path, "wb") as f:
                f.write(image_bytes)
            print(f"DEBUG [OpenAI]: Image saved to {local_path}")
            return str(local_path)
        
        # Handle URL response
        elif hasattr(image_data, 'url') and image_data.url:
            print(f"DEBUG [OpenAI]: Downloading image from URL...")
            image_response = requests.get(image_data.url, timeout=30)
            
            if image_response.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(image_response.content)
                print(f"DEBUG [OpenAI]: Image saved to {local_path}")
                return str(local_path)
            else:
                print(f"ERROR [OpenAI]: Failed to download image, status code: {image_response.status_code}")
                return None
        else:
            print(f"ERROR [OpenAI]: Response contains neither URL nor base64 data")
            return None
        
    except Exception as e:
        print(f"ERROR [OpenAI]: Exception occurred: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return None

class UserProfile(BaseModel):
    learning_style: Optional[Literal["active", "reflective", "analytical", "passive"]] = Field(..., description="The learning style of the user.")
    background_level: Optional[Literal["beginner", "intermediate", "advanced"]] = Field(..., description="The learning level of the user.")
    goal: Optional[Literal["learning", "job", "both"]] = Field(default=None, description="The ultimate goal for learning.")
    interests: Optional[List[str]] = Field(default_factory=list, description="The interests of the user.")
    topic: str = Field(..., description="The topic that user is interested in learning.")
    analysis: Optional[str] = Field(None, description="Personalized analysis of the user's profile and learning path.")

class Exercise(BaseModel):
    title: str = Field(..., description="Title of the exercise or example.")
    description: str = Field(..., description="Content or instructions. Use standard LaTeX for math (e.g., $E=mc^2$).")
    solution: Optional[str] = Field(None, description="The correct Python implementation or solution code.")
    image_urls: List[str] = Field(default_factory=list, description="A list of 2-3 highly relevant image URLs from the internet (e.g., Unsplash) that illustrate the concept.")
    sample_data: Optional[str] = Field(None, description="Clear, concise sample data in Python code format if required for the exercise.")

class KnowledgeGap(BaseModel):
    missing_concepts: List[str] = Field(..., description="The missing concepts of the user.")
    misconceptions: List[str] = Field(..., description="The common misconceptions of the user.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="The self-assessed confidence of the user.")
    analysis: Optional[str] = Field(None, description="Personalized explanation of the gaps and how we'll address them.")
    
class TeachingPlan(BaseModel):
    teaching_strategy: str = Field(..., description="The teaching strategy (active, reflective, analytical, passive).")
    lessons: List[str] = Field(..., description="The lessons to be taught.")
    examples: List[Exercise] = Field(..., description="Illustrative examples.")
    exercises: List[Exercise] = Field(..., description="Exercises for assessment.")
    context_notes: str = Field(..., description="Additional context/search snippets.") 
    
class LearningAssessment(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Assessment score.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Assessment confidence.")
    feedback: str = Field(..., description="Assessment feedback.")
    recommendations: str = Field(..., description="Assessment recommendations.")
    next_steps: Literal["reteach_theory", "reteach_examples", "reinforce", "advance", "None"] = Field(..., description="Assessment next steps.")
    error_type: Literal["conceptual", "application", "methodological"] = Field(..., description="Type of error.")
    analysis: Optional[str] = Field(None, description="Personalized assessment summary and encouragement.")
    next_step_analysis: Optional[str] = Field(None, description="Explanation for the recommended next step.")
    
class LearningMemory(BaseModel):
    completed_lessons: List[str] = Field(default_factory=list, description="Lessons already completed.")
    previous_scores: List[float] = Field(default_factory=list, description="Scores from past assessments.")
    notes: List[str] = Field(default_factory=list, description="Additional notes or reflections.")
    current_session_log: List[str] = Field(default_factory=list, description="Log of user interactions in the current session (code runs, quiz answers).")
    
class LearningState(BaseModel):
    user_profile: UserProfile
    knowledge_gap: KnowledgeGap
    teaching_plan: TeachingPlan
    learning_assessment: LearningAssessment
    learning_memory: LearningMemory
    
class LearningExercises(BaseModel):
    examples: List[Exercise] = Field(..., description="The examples to be used in the lessons.")
    exercises: List[Exercise] = Field(..., description="The assessments to be used to evaluate learning.")
    
llm_with_knowledge_gaps = llm.with_structured_output(KnowledgeGap)
llm_with_teaching_plan = llm.with_structured_output(TeachingPlan)
llm_with_learning_exercises = llm.with_structured_output(LearningExercises)
llm_with_learning_assessment = llm.with_structured_output(LearningAssessment)

def compile_user_profile(state: LearningState) -> UserProfile:
    """
        Dynamically update the user's profile by preserving their explicit selections
        and enriching with insights from learning history.
    """
    
    # Always preserve the user's current explicit selections
    current_profile = state.user_profile
    
    # Gather context for enrichment
    past_lessons = state.learning_memory.completed_lessons
    previous_scores = state.learning_memory.previous_scores
    current_knowledge_gaps = state.knowledge_gap.missing_concepts
    misconceptions = state.knowledge_gap.misconceptions

    # If no history yet, return the current profile as-is
    if not past_lessons and not previous_scores and not current_knowledge_gaps:
        return current_profile

    # Otherwise, enrich the profile with learning history insights
    # But PRESERVE the user's explicit selections for core fields
    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are an expert educational data analyst. "
        "Analyze the user's learning history to suggest additional interests or insights, "
        "but DO NOT change their explicitly selected learning style, background level, or goals."
    )
    
    human_prompt = HumanMessagePromptTemplate.from_template(
        """
        User's current profile:
        - Learning style: {learning_style}
        - Background level: {background_level}
        - Goal: {goal}
        - Current interests: {interests}
        - Topic: {topic}

        Learning history:
        - Completed lessons: {completed_lessons}
        - Previous assessment scores: {previous_scores}
        - Knowledge gaps: {knowledge_gaps}
        - Misconceptions: {misconceptions}

        Task:
        Based on the learning history, suggest any ADDITIONAL interests that might help the user.
        DO NOT change their learning_style, background_level, or goal - these are user preferences.
        
        Return the result in JSON format exactly like this:
        {{
            "learning_style": "...",
            "background_level": "...",
            "goal": "...",
            "interests": ["...", "..."],
            "analysis": "A warm, personalized message analyzing their profile (approx 50 words). THIS FIELD IS REQUIRED."
        }}
        """
    )
    
    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    
    formatted_prompt = prompt.format(
        learning_style=current_profile.learning_style,
        background_level=current_profile.background_level,
        goal=current_profile.goal,
        interests=current_profile.interests,
        topic=current_profile.topic,
        completed_lessons=past_lessons,
        previous_scores=previous_scores,
        knowledge_gaps=current_knowledge_gaps,
        misconceptions=misconceptions
    )
    
    response = llm.invoke(formatted_prompt).content
    
    try:
        # Parse full JSON response
        profile_data = json.loads(response)
        
        # Extract additional interests
        additional_interests = profile_data.get("interests", [])
        analysis_text = profile_data.get("analysis", "")
        
        if isinstance(additional_interests, list):
            # Merge with existing interests, avoiding duplicates
            all_interests = list(set(current_profile.interests + additional_interests))
        else:
            all_interests = current_profile.interests
            
    except (json.JSONDecodeError, TypeError):
        # If parsing fails, keep current interests
        all_interests = current_profile.interests
        analysis_text = "Analysis unavailable."

    if not analysis_text:
        analysis_text = "Profile updated based on your recent activity."
    
    # Create updated profile preserving user's core selections
    updated_profile = UserProfile(
        learning_style=current_profile.learning_style,  # PRESERVE
        background_level=current_profile.background_level,  # PRESERVE
        goal=current_profile.goal,  # PRESERVE
        interests=all_interests,  # ENHANCE
        topic=current_profile.topic,  # PRESERVE
        analysis=analysis_text # NEW
    )
    
    return updated_profile
    
def detect_knowledge_gaps(state: LearningState) -> KnowledgeGap:
    """
        Dynamically detect user's knowledge gaps, misconceptions, and confidence.
        Uses:
        - User's background level and topic
        - Past assessment scores
        - Completed lessons
        - Learning memory notes
        - Interests
        - Current learning plan
    """
    # Gather context from state
    background_level = state.user_profile.background_level
    topic = state.user_profile.topic
    completed_lessons = state.learning_memory.completed_lessons
    previous_scores = state.learning_memory.previous_scores
    notes = state.learning_memory.notes
    interests = state.user_profile.interests
    teaching_materials = state.teaching_plan.lessons or ""
    current_examples = state.teaching_plan.examples or ""
    
    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are an expert educational diagnostician. Analyze a learner's "
        "background level, topic of interest, history, notes, and completed lessons "
        "to infer knowledge gaps and misconceptions RELEVANT to their current topic and level."
    )

    human_prompt = HumanMessagePromptTemplate.from_template(
        """
        Learner profile:
        - Background level: {background_level}
        - Topic of study: {topic}
        - Interests: {interests}
        
        Learning history:
        - Completed lessons: {completed_lessons}
        - Previous scores: {previous_scores}
        - Notes & reflections: {notes}
        - Teaching materials: {teaching_materials}

        Task:
        Analyze the learner who is at {background_level} level and studying {topic}.
        
        1. Identify SPECIFIC missing concepts relevant to {topic} that are appropriate for their {background_level} level.
           - For beginners: focus on foundational concepts
           - For intermediate: focus on deeper understanding and connections
           - For advanced: focus on nuanced details and edge cases
        
        2. Identify misconceptions or common misunderstandings specific to {topic}.
        
        3. Estimate learner's confidence level (0.0 low - 1.0 high) based on their background level:
           - Beginners typically start at 0.2-0.4
           - Intermediate learners start at 0.4-0.6
           - Advanced learners start at 0.6-0.8

        4. Provide a personalized analysis of these gaps: talk directly to the learner, explain WHY they might be struggling with these specific concepts, and offer encouragement.

        IMPORTANT: Gaps should be SPECIFIC to {topic}, not generic concepts like "Fundamentals of Data Science".

        Return in JSON format exactly like this:
        {{
            "missing_concepts": ["..."],
            "misconceptions": ["..."],
            "confidence": float,
            "analysis": "Personalized explanation..."
        }}
        """
    )

    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    formatted_prompt = prompt.format(
        background_level=background_level,
        topic=topic,
        completed_lessons=completed_lessons,
        previous_scores=previous_scores,
        notes=notes,
        interests=interests,
        teaching_materials=teaching_materials,
        current_examples=current_examples
    )

    response = llm_with_knowledge_gaps.invoke(formatted_prompt)
    if not response.analysis:
        response.analysis = "Gaps identified based on your profile and history."
    return response

def extract_snippets(hits, max_items=5):
    """
    Extract content from search results, handle strings and dicts.
    Returns a list of concise snippets.
    """
    if isinstance(hits, str):
        hits = [{"title": "Search Result", "content": hits}]
    
    if not isinstance(hits, list):
        return []

    snippets = []
    for item in hits[:max_items]:
        if not isinstance(item, dict):
            if isinstance(item, str):
                item = {"title": "Info Block", "content": item}
            else:
                continue

        title = item.get("title") or item.get("name") or "Source"
        content = item.get("content") or item.get("snippet") or "" 
        
        if content:
            summary_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template("Summarize concisely for a learner."),
                HumanMessagePromptTemplate.from_template("Title: {title}\nContent: {content}")
            ])
            try:
                summary = llm.invoke(summary_prompt.format(title=title, content=content)).content
            except Exception:
                summary = content[:200] + "..."
        else:
            summary = "No detailed content available."
        
        link = item.get("link") or item.get("url") or "No link provided"
        snippets.append(f"{title}: {summary} (Link: {link})")
    
    return snippets

def build_citations(tavily_hits, arxiv_hits, gscholar_hits, youtube_hits):
    """Deterministically build a Citations section from search results."""
    lines = ["\n\n## 📚 Citations & References\n"]
    seen_links = set()
    
    # 1. Process Web/Tavily (List of Dicts)
    if tavily_hits and isinstance(tavily_hits, list):
        lines.append("\n### 🌐 Web Sources")
        for hit in tavily_hits:
            if isinstance(hit, dict):
                url = hit.get('url') or hit.get('link')
                title = hit.get('title', 'Reference')
                if url and url not in seen_links:
                    lines.append(f"- [{title}]({url})")
                    seen_links.add(url)

    # 2. Process Google Scholar (Often returns string summary)
    if gscholar_hits:
        lines.append("\n### 🎓 Google Scholar")
        if isinstance(gscholar_hits, str):
            # Regex to find http links in the text
            found_links = re.findall(r'(https?://[^\s]+)', gscholar_hits)
            if found_links:
                for link in found_links:
                     # Clean trailing punctuation
                     link = link.rstrip('.,;)')
                     if link not in seen_links:
                         lines.append(f"- [Scholar Link]({link})")
                         seen_links.add(link)
            else:
                 # If no links, just show the summary as blockquote
                 clean_scholar = gscholar_hits.replace("Summary:", "").strip()
                 lines.append(f"> {clean_scholar}")
        elif isinstance(gscholar_hits, list): 
             for hit in gscholar_hits:
                 lines.append(f"- {hit}")

    # 3. Process Arxiv (Often returns string summary)
    if arxiv_hits:
        lines.append("\n### 📄 Arxiv Papers")
        if isinstance(arxiv_hits, str):
             # 1. Try to find explicit arXiv IDs (format: arXiv:1706.03762)
             import re
             arxiv_ids = re.findall(r'arXiv:(\d+\.\d+)', arxiv_hits)
             for aid in arxiv_ids:
                 url = f"https://arxiv.org/abs/{aid}"
                 if url not in seen_links:
                     lines.append(f"- [Arxiv Paper {aid}]({url})")
                     seen_links.add(url)
            
             # 2. Also look for full links
             found_links = re.findall(r'(https?://arxiv\.org/abs/\d+\.\d+)', arxiv_hits)
             for link in found_links:
                 if link not in seen_links:
                     lines.append(f"- [Arxiv Link]({link})")
                     seen_links.add(link)
             
             # If no links found, show summary text (fallback)
             if not arxiv_ids and not found_links:
                 clean_arxiv = arxiv_hits.replace("Summary:", "").strip()
                 lines.append(f"> {clean_arxiv}")

        elif isinstance(arxiv_hits, list):
            for hit in arxiv_hits:
                if isinstance(hit, dict):
                    url = hit.get('url') or hit.get('link')
                    if url and url not in seen_links:
                         lines.append(f"- {hit.get('title', 'Paper')}: {url}")
                         seen_links.add(url)

    # 4. Process YouTube (Tool returns string representation of list)
    if youtube_hits:
        lines.append("\n### 📺 Video Tutorials")
        urls = []
        if isinstance(youtube_hits, str):
            try:
                # Output format: "['link1', 'link2']"
                if "[" in youtube_hits and "]" in youtube_hits:
                    urls = ast.literal_eval(youtube_hits)
                elif "http" in youtube_hits:
                    urls = [youtube_hits]
            except:
                pass
        elif isinstance(youtube_hits, list):
            urls = youtube_hits
            
        for url in urls:
            if isinstance(url, str) and url.startswith("http") and url not in seen_links:
                lines.append(f"- [Watch Video]({url})")
                seen_links.add(url)

    return "\n".join(lines)

def generate_lesson_content(state: LearningState) -> Dict:
    """Subgraph Node: Generate core lesson content."""
    user_profile = state.user_profile.model_dump()
    knowledge_gaps = state.knowledge_gap.model_dump()
    
    search_query = "Explain " + user_profile["topic"]
    
    tavily_hits = tavily_search_tool.run(search_query) if tavily_search_tool else []
    
    # Arxiv Search
    arxiv_hits = []
    try:
        arxiv_query = user_profile["topic"]
        print(f"DEBUG [Arxiv]: Searching for '{arxiv_query}'...")
        
        search = arxiv.Search(
            query=arxiv_query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        for paper in search.results():
            arxiv_hits.append({
                "title": paper.title,
                "content": paper.summary,
                "url": paper.entry_id,
                "link": paper.entry_id
            })
        print(f"DEBUG [Arxiv]: Found {len(arxiv_hits)} papers.")

    except Exception as e:
        print(f"ERROR [Arxiv]: Failed to load papers: {e}")
        import traceback
        traceback.print_exc()
        arxiv_hits = []

        
    # Robust Google Scholar Search
    try:
        gscholar_hits = google_scholar_search_tool.run(search_query) if google_scholar_search_tool else []
    except Exception:
        gscholar_hits = []
        
    youtube_hits = youtube_search_tool.run(search_query) if youtube_search_tool else []
    
    combined_snippets = (extract_snippets(tavily_hits) + extract_snippets(arxiv_hits) + 
                         extract_snippets(gscholar_hits) + extract_snippets(youtube_hits))
    
    lesson_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a World-Class Technical Educator and Domain Expert. Your goal is to provide a State-of-the-Art (SOTA) learning experience. "
            "Follow these strict pedagogical principles:\n"
            "1. FIRST-PRINCIPLES: Do not start with definitions. Start with the fundamental problem or 'ground truth' that necessitated this topic. Build up from there.\n"
            "2. DIRECT EDUCATION: Speak directly to the user. No meta-talk, no outlines, no objectives. Provide the raw, deep knowledge immediately.\n"
            "3. SOCRATIC ENGAGEMENT: Periodically embed a 'Pause & Think' question (in a blockquote) that asks the user to predict a result or reason through a concept before you explain it.\n"
            "4. CONCEPTUAL DEPTH: Do not skim the surface. Explain the 'why' behind every mechanism. Use high-quality analogies that bridge the gap between abstract math/logic and real-world intuition.\n"
            "5. VISUAL CLARITY: Use ASCII diagrams, structured tables, or flowcharts. Use Unicode for Greek letters and SUBSCRIPTS where available (e.g., β₀, β₁, λ, σ, Σ). "
            "   IMPORTANT: For simple variables like beta_0, always use β₀. DO NOT use LaTeX for single variables unless part of a complex formula.\n"
            "6. MATH FORMATTING: Use single $ for inline math (e.g., $x₁$) and double $$ for standalone block equations. DO NOT use \\[ or \\].\n"
            "7. ADAPTIVE SCAFFOLDING: Use the detected knowledge gaps to provide extra detail where needed.\n"
            "8. UNIVERSAL ACCESSIBILITY: Use clear language and relatable examples.\n"
            "9. ILLUSTRATIONS: You MUST include at least 2-3 descriptive placeholders for images using Markdown syntax: ![Brief context-aware description](https://source.unsplash.com/featured/?topic_keyword)."
        ),
        HumanMessagePromptTemplate.from_template(
            "Topic: {topic}\n"
            "Learning Style: {learning_style}\n"
            "Knowledge Gaps: {knowledge_gaps}\n"
            "Source Material:\n{snippets}\n\n"
            "Task: Write the definitive, deep-dive educational chapter for this topic. Use Unicode Greek letters (β, α, etc.) and subscripts (₀, ₁, ₂) for variables."
        )
    ])
    
    formatted_prompt = lesson_prompt.format(
        topic=user_profile["topic"],
        learning_style=user_profile["learning_style"],
        knowledge_gaps=knowledge_gaps.get("missing_concepts", []),
        snippets="\n".join(combined_snippets)
    )
    
    print(f"DEBUG [Lesson]: Generating content for {user_profile['topic']}...")
    lesson_content = llm.invoke(formatted_prompt).content
    print(f"DEBUG [Lesson]: Content generated ({len(lesson_content)} chars).")
    
    # Append Citations
    citations_md = build_citations(tavily_hits, arxiv_hits, gscholar_hits, youtube_hits)
    lesson_content += citations_md
    
    state.teaching_plan.lessons = [lesson_content]
    state.teaching_plan.context_notes = "\n".join(combined_snippets)
    state.teaching_plan.teaching_strategy = user_profile["learning_style"]
    
    return {"teaching_plan": state.teaching_plan}

def generate_visual_aids(state: LearningState) -> Dict:
    """Subgraph Node: Generate custom illustrations using OpenAI and localize images."""
    topic = state.user_profile.topic
    lesson_content = state.teaching_plan.lessons[0]
    
    # 1. Use OpenAI Image API to replace first two image placeholders
    pattern = r"!\[(.*?)\]\((.*?)\)"
    openai_count = 0
    
    def openai_replace(match):
        nonlocal openai_count
        alt_text = match.group(1)
        original_url = match.group(2)
        
        if openai_count < 3:
            print(f"DEBUG [Visuals]: Generating OpenAI image for '{alt_text}'...")
            # Pass alt_text as context for meaningful filename
            local_path = generate_openai_image(
                prompt=f"Educational illustration of {alt_text}, professional digital art, high resolution, {topic}",
                context=alt_text
            )
            if local_path:
                openai_count += 1
                return f"![{alt_text}]({local_path})"
        return match.group(0) # Keep original for process_lesson_images to handle
        
    lesson_content = re.sub(pattern, openai_replace, lesson_content)
    
    # 2. Localize any remaining external images
    print("DEBUG [Visuals]: Localizing external images...")
    lesson_content = process_lesson_images(lesson_content)
    
    state.teaching_plan.lessons[0] = lesson_content
    print(f"DEBUG [Visuals]: Finished. Generated {openai_count} OpenAI images.")
    
    return {"teaching_plan": state.teaching_plan}

def generate_examples_exercises(state: LearningState) -> Dict:
    """Subgraph Node: Generate practical examples and exercises."""
    lesson_content = state.teaching_plan.lessons[0]
    
    exercises_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a Senior Technical Architect and Mentor. Your task is to provide rigorous practical implementation and assessment. "
            "Follow these principles:\n"
            "1. PRODUCTION-GRADE CODE: Examples must be PEP8 compliant, well-commented.\n"
            "2. BLOOM'S TAXONOMY: Design 5 exercises that progress in difficulty. YOU MUST PROVIDE EXACTLY 1 EXAMPLE AND 5 EXERCISES.\n"
            "3. PREMIUM FORMATTING: Code implementations MUST be enclosed in ```python markdown blocks. Exercises MUST be formatted in clean, standard Markdown. Use Unicode for math (β₀, β₁ etc.). Each exercise title should be just the title text.\n"
            "4. RICH CONTENT: Provide 2-3 relevant, verified `image_urls` for each example and exercise. If direct URLs are unavailable, use https://source.unsplash.com/featured/?[topic_keyword] as a reliable fallback. Include `sample_data` (as code block) if the exercise requires data input."
        ),
        HumanMessagePromptTemplate.from_template(
            "Lesson Content:\n{lesson}\n"
            "Task:\n"
            "1. Provide a 'Masterclass' Code Implementation. Place the full code in the `solution` field of the example.\n"
            "2. Provide 5 challenging 'Level-Up' Exercises. Each must have its full solution code, 2-3 `image_urls`, and `sample_data` if relevant.\n"
            "CRITICAL: Do not return empty lists."
        )
    ])
    
    print("DEBUG [Exercises]: Generating examples and exercises...")
    learning_exercises: LearningExercises = llm_with_learning_exercises.invoke(exercises_prompt.format(lesson=lesson_content))
    
    if not learning_exercises or (not learning_exercises.examples and not learning_exercises.exercises):
        print("DEBUG [Exercises]: ERROR - LLM returned empty exercises. Retrying with explicit reminder...")
        learning_exercises = llm_with_learning_exercises.invoke(exercises_prompt.format(lesson=lesson_content) + "\nREMINDER: Examples and Exercises lists MUST NOT be empty.")

    # Process examples and exercises to download images locally
    for i, ex in enumerate(learning_exercises.examples):
        local_paths = []
        for j, url in enumerate(ex.image_urls):
            local_path = download_image(url, f"example_{i}_{j}")
            if local_path:
                local_paths.append(local_path)
        ex.image_urls = local_paths
                
    for i, ex in enumerate(learning_exercises.exercises):
        local_paths = []
        for j, url in enumerate(ex.image_urls):
            local_path = download_image(url, f"exercise_{i}_{j}")
            if local_path:
                local_paths.append(local_path)
        ex.image_urls = local_paths
    
    state.teaching_plan.examples = learning_exercises.examples
    state.teaching_plan.exercises = learning_exercises.exercises
    print(f"DEBUG [Exercises]: Finished. Examples: {len(learning_exercises.examples)}, Exercises: {len(learning_exercises.exercises)}")
    
    return {"teaching_plan": state.teaching_plan}

# Teaching Plan Subgraph
tp_builder = StateGraph(LearningState)
tp_builder.add_node("generate_lesson_content", generate_lesson_content)
tp_builder.add_node("generate_visual_aids", generate_visual_aids)
tp_builder.add_node("generate_examples_exercises", generate_examples_exercises)

tp_builder.add_edge(START, "generate_lesson_content")
tp_builder.add_edge("generate_lesson_content", "generate_visual_aids")
tp_builder.add_edge("generate_visual_aids", "generate_examples_exercises")
tp_builder.add_edge("generate_examples_exercises", END)

teaching_plan_subgraph = tp_builder.compile()

def generate_teaching_plan(state: LearningState) -> Dict:
    """Wrapper node for the teaching plan subgraph."""
    state_dict = state.model_dump()
    result = teaching_plan_subgraph.invoke(state_dict)
    return result

def assess_learning(state: LearningState) -> LearningAssessment:
    """
        Evaluate the learner's understanding based on the teaching plan, completed lessons, and exercises.
        
        Steps:
        1) Take the generated lessons, examples, and exercises from the TeachingPlan.
        2) Use the learner's past performance and notes to adapt assessment difficulty.
        3) Generate a structured assessment with score, confidence, feedback, and recommendations.
    """
    # Gather context
    teaching_plan = state.teaching_plan
    completed_lessons = state.learning_memory.completed_lessons
    previous_scores = state.learning_memory.previous_scores
    notes = state.learning_memory.notes
    current_log = state.learning_memory.current_session_log
    knowledge_gaps = state.knowledge_gap.model_dump()
    
    # Prepare assessment prompt
    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are an expert educational assessor. Evaluate a learner's understanding, confidence, and misconceptions based on their lessons and exercises."
    )
    
    human_prompt = HumanMessagePromptTemplate.from_template(
        """
        Learner context:
        - Completed lessons: {completed_lessons}
        - Past scores: {previous_scores}
        - Notes & reflections: {notes}
        - Current Session Activity (Recent Code Runs/Interactions): {current_log}
        - Knowledge gaps: {knowledge_gaps}

        Teaching plan:
        - Lessons: {lessons}
        - Examples: {examples}
        - Exercises: {exercises}

        Task:
        1. Assign a score (0.0 to 1.0) reflecting mastery of the lesson.
        2. Estimate the learner's confidence level (0.0 to 1.0).
        3. Identify errors type: conceptual, application, or methodological.
        4. Provide detailed feedback on strengths and weaknesses.
        5. Recommend next steps and further exercises.
        6. Provide a personalized analysis of their performance: be encouraging but honest. Mention specific errors or successes from their session log.

        Return in JSON format exactly like this:
        {{
            "score": float,
            "confidence": float,
            "error_type": "conceptual|application|methodological",
            "feedback": "...",
            "recommendations": "...",
            "recommendations": "...",
            "next_steps": "reteach_theory|reteach_examples|reinforce|advance",
            "analysis": "Personalized assessment..."
        }}
        """
    )
    
    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    
    formatted_prompt = prompt.format(
        completed_lessons=completed_lessons,
        previous_scores=previous_scores,
        notes=notes,
        knowledge_gaps=knowledge_gaps,
        lessons=teaching_plan.lessons,
        examples=teaching_plan.examples,
        exercises=teaching_plan.exercises,
        current_log=current_log
    )
    
    learning_assessment = llm_with_learning_assessment.invoke(formatted_prompt)
    if not learning_assessment.analysis:
        learning_assessment.analysis = "Assessment completed based on your performance."
        
    # Update Learning Memory with new stats
    if learning_assessment.score is not None:
        state.learning_memory.previous_scores.append(learning_assessment.score)
        
    # Archive session to notes
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    note_content = f"[{timestamp}] Score: {learning_assessment.score:.2f}, Conf: {learning_assessment.confidence:.2f}. {learning_assessment.feedback[:50]}..."
    state.learning_memory.notes.append(note_content)
    
    # Clear processed logs so next assessment is fresh
    state.learning_memory.current_session_log = []
    
    return learning_assessment

async def evaluate(state: LearningState):
    """
    Decide the next step based on assessment.
    Updates the `next_steps` field in LearningAssessment.
    """
    assessment: LearningAssessment = state.learning_assessment

    if assessment.score is None or assessment.confidence is None:
        # fallback if assessment is incomplete
        next_step = "reteach_theory"
    elif assessment.score >= 0.75 and assessment.confidence >= 0.7:
        next_step = "advance"
    elif assessment.score >= 0.75:
        next_step = "reinforce"
    elif assessment.error_type == "conceptual":
        next_step = "reteach_theory"
    else:
        next_step = "reteach_examples"

    # Generate analysis for the next step using LLM
    analysis_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a personalized learning mentor. Explain why a specific next step was chosen for the learner."
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Learner Assessment:
            - Score: {score}
            - Confidence: {confidence}
            - Feedback: {feedback}
            
            Chosen Next Step: {next_step}
            
            Task:
            Write a short, encouraging message (approx 30-50 words) explaining WHY this next step is the best specific action for them properly.
            Do NOT repeat the assessment feedback. Focus on the FUTURE action.
            """
        )
    ])
    
    formatted_prompt = analysis_prompt.format(
        score=assessment.score,
        confidence=assessment.confidence,
        feedback=assessment.feedback,
        next_step=next_step
    )
    
    next_step_analysis = llm.invoke(formatted_prompt).content
    
    # Update assessment with decision and analysis
    assessment.next_steps = next_step
    assessment.next_step_analysis = next_step_analysis
    state.learning_assessment = assessment # Update the state with the modified assessment

    # Ensure assessment has all required fields before returning
    if not assessment.next_steps:
        assessment.next_steps = "advance"

    return {"learning_assessment": assessment}

# Next step handlers
# - Re-teach theory
# - Advance
# - Reinforce
# - Re-teach examples

async def reteach_theory(state: LearningState):
    """
    Dynamically generate additional theory-based content for a learner.
    Uses the learner's profile, knowledge gaps, and current lesson.
    """
    user_profile = state.user_profile
    knowledge_gaps = state.knowledge_gap
    current_lesson = state.teaching_plan.lessons[-1] if state.teaching_plan.lessons else "Lesson content"

    # Prompt the LLM to generate a re-teaching section focused on knowledge gaps
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an Elite Technical Mentor. Your task is 'Deep Adaptive Feedback'. "
            "When a learner is confused, don't just repeat the facts. Identify the likely ROOT misconception and dismantle it. "
            "Use first-principles reasoning and crystal-clear analogies."
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Topic: {lesson}
            Confused Concepts: {missing_concepts}
            Likely Misconceptions: {misconceptions}

            Task:
            1. Conduct a 'Misconception Takedown': Explain why the student's current mental model might be slightly off.
            2. Re-explain from First Principles: Ground the concepts in simple, unassailable logic.
            3. Predictive Challenge: Provide a 'Check for Understanding' question at the end.
            """
        )
    ])

    formatted_prompt = prompt.format(
        learning_style=user_profile.learning_style,
        background_level=user_profile.background_level,
        interests=", ".join(user_profile.interests),
        lesson=current_lesson,
        missing_concepts=", ".join(knowledge_gaps.missing_concepts),
        misconceptions=", ".join(knowledge_gaps.misconceptions)
    )

    new_theory = llm.invoke(formatted_prompt).content
    state.teaching_plan.lessons.append(new_theory)

    return {"teaching_plan": state.teaching_plan, "execution_result": new_theory}


async def reteach_examples(state: LearningState):
    """
    Dynamically generate additional illustrative examples to reinforce understanding.
    Focuses on the learner's weak areas identified in knowledge gaps.
    """
    user_profile = state.user_profile
    knowledge_gaps = state.knowledge_gap
    current_lesson = state.teaching_plan.lessons[-1] if state.teaching_plan.lessons else "Lesson content"

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an expert educator generating tailored examples for a learner."
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Learner profile:
            - Learning style: {learning_style}
            - Background level: {background_level}

            Current lesson:
            {lesson}

            Knowledge gaps:
            - Missing concepts: {missing_concepts}
            - Misconceptions: {misconceptions}

            Task:
            Generate 3-5 additional illustrative examples or mini-exercises that:
            1. Directly address the missing concepts.
            2. Correct common misconceptions.
            3. Are adapted to the learner's learning style.
            Provide the examples in clear, concise language.
            """
        )
    ])

    formatted_prompt = prompt.format(
        learning_style=user_profile.learning_style,
        background_level=user_profile.background_level,
        lesson=current_lesson,
        missing_concepts=", ".join(knowledge_gaps.missing_concepts),
        misconceptions=", ".join(knowledge_gaps.misconceptions)
    )

    new_examples_text = llm.invoke(formatted_prompt).content
    state.teaching_plan.examples.append(Exercise(title="Additional Example", description=new_examples_text))

    return {"teaching_plan": state.teaching_plan, "execution_result": new_examples_text}


async def reinforce(state: LearningState):
    """
    Reinforce a lesson by:
    - Summarizing key takeaways
    - Creating mini-exercises
    - Updating memory
    """
    lesson_content = state.teaching_plan.lessons[-1] if state.teaching_plan.lessons else "Lesson content"
    user_profile = state.user_profile

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a Technical Coach specializing in high-performance learning. Your goal is to maximize long-term retention. "
            "Identify the 'Golden Nuggets' (highest-ROI concepts) and provide 'Deep-End' challenges."
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Topic: {lesson}
            Learning Style: {learning_style}

            Task:
            1. GOLDEN NUGGETS: Distill the entire lesson into 3-5 maximum-impact takeaways that the learner must NEVER forget.
            2. DEEP-END CHALLENGE: Provide two 'level-infinite' coding challenges that require the learner to synthesize everything they've learned with external logic.
            3. MENTAL MAP: Using only text/ASCII, describe how this concept fits into the broader ecosystem of its field.
            """
        )
    ])

    formatted_prompt = prompt.format(
        learning_style=user_profile.learning_style,
        background_level=user_profile.background_level,
        lesson=lesson_content
    )

    reinforcement_content = llm.invoke(formatted_prompt).content

    # Update exercises and memory
    state.teaching_plan.exercises.append(Exercise(title="Reinforcement Exercise", description=reinforcement_content))
    state.learning_memory.completed_lessons.append(lesson_content)

    return {"teaching_plan": state.teaching_plan, "learning_memory": state.learning_memory, "execution_result": reinforcement_content}


async def advance(state: LearningState):
    """
    Advance the learner by:
    - Marking lesson as completed
    - Suggesting next content based on profile and interests
    """
    user_profile = state.user_profile
    lesson_content = state.teaching_plan.lessons[-1] if state.teaching_plan.lessons else "Lesson content"

    # Generate suggestion for next lesson or follow-up content
    # Generate graduation/advancement message
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a motivational mentor congratulating a learner on mastering a topic."
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Learner profile:
            - Goal: {goal}
            - Interests: {interests}

            Topic Mastered: {lesson}
            Performance: Score {score}

            Task:
            1. Congratulate the learner enthusiastically.
            2. Highlight their progress.
            3. Recommend 3 specific external platforms (e.g., Kaggle, LeetCode, GitHub projects) tailored to their interests where they can apply this skill.
            4. Keep it inspiring and actionable.
            """
        )
    ])
    
    # Get latest score if available
    latest_score = state.learning_memory.previous_scores[-1] if state.learning_memory.previous_scores else "N/A"

    formatted_prompt = prompt.format(
        goal=user_profile.goal,
        interests=", ".join(user_profile.interests),
        lesson=lesson_content,
        score=latest_score
    )
    
    advance_content = llm.invoke(formatted_prompt).content
    state.learning_memory.completed_lessons.append(lesson_content)
    
    return {"learning_memory": state.learning_memory, "execution_result": advance_content}

    next_lesson_suggestion = llm.invoke(formatted_prompt).content

    state.learning_memory.completed_lessons.append(lesson_content)
    state.teaching_plan.lessons.append(next_lesson_suggestion)

    return {"teaching_plan": state.teaching_plan, "learning_memory": state.learning_memory}

# Initialize a graph
graph = StateGraph(LearningState)

# START → Compile user profile
graph.add_node("compile_user_profile", compile_user_profile)
graph.add_edge(START, "compile_user_profile")

# Compile profile → Detect knowledge gaps
graph.add_node("detect_knowledge_gaps", detect_knowledge_gaps)
graph.add_edge("compile_user_profile", "detect_knowledge_gaps")

# Detect gaps → Generate teaching plan (Subgraph)
graph.add_node("generate_teaching_plan", generate_teaching_plan)
graph.add_edge("detect_knowledge_gaps", "generate_teaching_plan")

# Teaching plan → Assess learning
graph.add_node("assess_learning", assess_learning)
graph.add_edge("generate_teaching_plan", "assess_learning")

# Assessment → Evaluate next step
graph.add_node("evaluate", evaluate)
graph.add_edge("assess_learning", "evaluate")

# Add nodes for all next-step tools
graph.add_node("reteach_theory", reteach_theory)
graph.add_node("reteach_examples", reteach_examples)
graph.add_node("reinforce", reinforce)
graph.add_node("advance", advance)

# Conditional edges from 'evaluate' → next step
# Route based on the assessment's next_steps
def route_to_next_step(state: LearningState) -> str:
    return state.learning_assessment.next_steps

graph.add_conditional_edges(
    "evaluate",
    route_to_next_step,
    {
        "reteach_theory": "reteach_theory",
        "reteach_examples": "reteach_examples",
        "reinforce": "reinforce",
        "advance": "advance"
    }
)

# After reteach/reinforce → reassess
for step in ["reteach_theory", "reteach_examples", "reinforce"]:
    graph.add_edge(step, "assess_learning")

# After advance → optionally END
graph.add_edge("advance", END)

# Compile the graph
learning_workflow = graph.compile()

# Save the graph visualization
try:
    with open("learning_workflow.png", "wb") as f:
        f.write(learning_workflow.get_graph().draw_mermaid_png())
    print("Workflow visualization saved to 'learning_workflow.png'")
except Exception as e:
    print(f"Could not save workflow visualization: {e}")

print("Learning workflow created successfully!")