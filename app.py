import streamlit as st
from pydantic import BaseModel
from typing import List
from datetime import datetime
import json, asyncio, sys, io, contextlib, traceback
from workflow import (
    LearningState,
    UserProfile,
    KnowledgeGap,
    TeachingPlan,
    LearningAssessment,
    LearningMemory,
    compile_user_profile,
    detect_knowledge_gaps,
    generate_teaching_plan,
    assess_learning,
    evaluate,
    reteach_theory,
    reteach_examples,
    reinforce,
    advance
)

# Set page config
st.set_page_config(
    page_title="Interactive Adaptive Learning",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set title and description
st.title("🧠 Adaptive Personalized Learning System")
st.markdown(
    """
This interactive platform simulates a personalized learning environment.
You can input your profile, explore lessons, complete exercises, receive assessments, 
and follow personalized next steps dynamically.
"""
)

# Sidebar: Collect user profile
st.sidebar.header("1️⃣ User Profile")

# Initialize Learning Memory FIRST to use its values in widgets
if "learning_state" not in st.session_state:
    st.session_state.learning_state = LearningState(
        user_profile=UserProfile(
            learning_style="active",
            background_level="beginner",
            goal="learning",
            interests=["Data Science", "Machine Learning"],
            topic="Time Series Forecasting"
        ),
        knowledge_gap=KnowledgeGap(missing_concepts=[], misconceptions=[], confidence=0.0),
        teaching_plan=TeachingPlan(teaching_strategy="active", lessons=[], examples=[], exercises=[], context_notes=""),
        learning_assessment=LearningAssessment(score=0.0, confidence=0.0, feedback="", recommendations="", next_steps="None", error_type="conceptual"),
        learning_memory=LearningMemory(completed_lessons=[], previous_scores=[], notes=[])
    )

ls_options = ["active", "reflective", "analytical", "passive"]
bl_options = ["beginner", "intermediate", "advanced"]
go_options = ["learning", "job", "both"]

# Sync Widgets with Session State
learning_style = st.sidebar.selectbox(
    "Preferred Learning Style",
    options=ls_options,
    index=ls_options.index(st.session_state.learning_state.user_profile.learning_style)
)

background_level = st.sidebar.selectbox(
    "Background Level",
    options=bl_options,
    index=bl_options.index(st.session_state.learning_state.user_profile.background_level)
)

goal = st.sidebar.selectbox(
    "Learning Goal",
    options=go_options,
    index=go_options.index(st.session_state.learning_state.user_profile.goal)
)

interests_str = st.sidebar.text_input(
    "Interests (comma separated)",
    value=", ".join(st.session_state.learning_state.user_profile.interests)
)

topic = st.sidebar.text_input(
    "Learning Topic",
    value=st.session_state.learning_state.user_profile.topic
)

# Update state immediately when inputs change
st.session_state.learning_state.user_profile.learning_style = learning_style
st.session_state.learning_state.user_profile.background_level = background_level
st.session_state.learning_state.user_profile.goal = goal
st.session_state.learning_state.user_profile.interests = [i.strip() for i in interests_str.split(",") if i.strip()]
st.session_state.learning_state.user_profile.topic = topic

learning_state: LearningState = st.session_state.learning_state

# Compile user profile dynamically
if st.button("🟢 Compile/Update User Profile"):
    updated_profile = compile_user_profile(learning_state)
    # Update session state directly to ensure persistence
    st.session_state.learning_state.user_profile = updated_profile
    learning_state.user_profile = updated_profile
    st.success("User profile compiled successfully!")
    if updated_profile.analysis:
        st.info(f"💡 **Analysis:** {updated_profile.analysis}")
    else:
        st.info("💡 **Analysis:** Profile updated based on your inputs.")  

# Detect knowledge gaps
if st.button("🔍 Detect Knowledge Gaps"):
    knowledge_gaps = detect_knowledge_gaps(learning_state)
    learning_state.knowledge_gap = knowledge_gaps
    st.success("Knowledge gaps detected!")
    if knowledge_gaps.analysis:
        st.warning(f"💡 **Gap Analysis:** {knowledge_gaps.analysis}")
    else:
        st.warning("💡 **Gap Analysis:** Gaps identified based on your profile.") 

# Generate teaching plan
if st.button("📚 Generate Teaching Plan"):
    with st.spinner("Generating personalized teaching plan..."):
        result = generate_teaching_plan(learning_state)
        if isinstance(result, dict) and "teaching_plan" in result:
             learning_state.teaching_plan = result["teaching_plan"]
        elif hasattr(result, "teaching_plan"):
             learning_state.teaching_plan = result.teaching_plan
        else:
             learning_state.teaching_plan = result
        st.success("Teaching plan generated!")

# Display teaching plan
teaching_plan = learning_state.teaching_plan
if teaching_plan and teaching_plan.lessons:
    st.divider()
    st.header("📖 Your Personalized Learning Journey")
    
    st.markdown("### 📝 Lessons")
    for idx, lesson in enumerate(teaching_plan.lessons, 1):
        with st.expander(f"Lesson {idx}", expanded=True):
            # Extract and render images from markdown
            import re
            from pathlib import Path
            
            # Split lesson content by image markdown patterns
            pattern = r'!\[(.*?)\]\((.*?)\)'
            parts = re.split(pattern, lesson)
            
            # Render content with images
            i = 0
            while i < len(parts):
                if i % 3 == 0:
                    # Regular text content
                    if parts[i].strip():
                        st.markdown(parts[i])
                elif i % 3 == 2:
                    # Image path (i-1 is alt text, i is path)
                    alt_text = parts[i-1] if i > 0 else "Image"
                    img_path = parts[i]
                    
                    # Check if image file exists
                    if Path(img_path).exists():
                        st.image(img_path, caption=alt_text, use_container_width=True)
                    else:
                        # If local file doesn't exist, try as URL
                        st.markdown(f"![{alt_text}]({img_path})")
                i += 1
            
    st.markdown("### 💡 Practical Examples")
    if teaching_plan.examples:
        for ex in teaching_plan.examples:
            with st.expander(f"📘 {ex.title}", expanded=True):
                if ex.image_urls:
                    for img_url in ex.image_urls:
                        # Check if image file exists locally
                        if Path(img_url).exists():
                            st.image(img_url, caption=f"Illustration for {ex.title}", use_container_width=True)
                        else:
                            st.markdown(f"![Illustration for {ex.title}]({img_url})")
                st.markdown(ex.description)
                if ex.solution:
                    st.markdown("#### 💻 Implementation")
                    st.code(ex.solution)
    else:
        st.info("No examples available.")

    st.markdown("## 📝 Practical Exercises")
    if teaching_plan.exercises:
        for idx, ex in enumerate(teaching_plan.exercises, 1):
            with st.container():
                # Clean title to prevent "Exercise 1: Exercise 1: ..." duplication
                clean_title = re.sub(r'^Exercise\s+\d+[:\.]\s*', '', ex.title, flags=re.IGNORECASE)
                st.markdown(f"### Exercise {idx}: {clean_title}")
                if ex.image_urls:
                    for img_url in ex.image_urls:
                        # Check if image file exists locally
                        if Path(img_url).exists():
                            st.image(img_url, caption=f"Illustration for {ex.title}", use_container_width=True)
                        else:
                            st.markdown(f"![Illustration for {ex.title}]({img_url})")
                st.markdown(ex.description)
                
                if ex.sample_data:
                    st.markdown("#### 📊 Sample Data")
                    st.code(ex.sample_data)

                # Code Playground
                code_input = st.text_area(
                    f"Your Implementation (Exercise {idx})", 
                    value="# Write your code here...",
                    height=200,
                    key=f"code_input_{idx}"
                )
                
                if st.button(f"▶️ Run Exercise {idx}", key=f"run_btn_{idx}"):
                    output_buffer = io.StringIO()
                    try:
                        import subprocess
                        import re
                        
                        # Combine sample_data and user code
                        sample_code = ex.sample_data or ""
                        
                        # Robust cleanup of markdown code blocks
                        def clean_code_block(code):
                            # Remove ```python and ``` lines
                            code = re.sub(r'^```\w*\s*$', '', code, flags=re.MULTILINE)
                            return code

                        sample_code = clean_code_block(sample_code)
                        user_code_cleaned = clean_code_block(code_input)

                        full_code = sample_code + "\n" + user_code_cleaned
                        
                        # Extract all import statements and separate them from the rest
                        lines = full_code.split('\n')
                        import_lines = []
                        code_lines = []
                        
                        for line in lines:
                            stripped = line.strip()
                            if stripped.startswith('import ') or stripped.startswith('from '):
                                import_lines.append(line)
                            elif stripped:  # Skip empty lines
                                code_lines.append(line)
                        
                        # Extract package names from imports for installation
                        import_pattern = r'^(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))'
                        all_imports = '\n'.join(import_lines)
                        imports = re.findall(import_pattern, all_imports, re.MULTILINE)
                        
                        # Map common module names to package names
                        package_map = {
                            'sklearn': 'scikit-learn',
                            'cv2': 'opencv-python',
                            'PIL': 'pillow',
                            'np': 'numpy',
                            'pd': 'pandas',
                            'plt': 'matplotlib',
                            'sns': 'seaborn',
                            'tf': 'tensorflow',
                            'torch': 'torch',
                            'keras': 'keras'
                        }
                        
                        # Get unique package names
                        packages = set()
                        for from_import, direct_import in imports:
                            module = from_import or direct_import
                            # Get base module name (e.g., 'sklearn' from 'sklearn.metrics')
                            base_module = module.split('.')[0]
                            # Skip built-in modules
                            if base_module not in ['builtins', 'sys', 'os', 're', 'io', 'json', 'traceback', 'contextlib', 'subprocess']:
                                # Map to correct package name
                                package_name = package_map.get(base_module, base_module)
                                packages.add(package_name)
                        
                        # Auto-install missing packages with clear feedback
                        if packages:
                            with st.status(f"📦 Setting up environment..."):
                                for package in sorted(packages):
                                    try:
                                        # Check if package is installed by trying to import it
                                        test_module = package.replace('-', '_')
                                        __import__(test_module)
                                        st.write(f"✅ {package} (already installed)")
                                    except ImportError:
                                        # Log failure for assessment
                                        log_entry = f"Exercise {idx} ('{ex.title}') execution: FAILED (ImportError for {package})."
                                        if not hasattr(st.session_state.learning_state.learning_memory, 'current_session_log'):
                                             st.session_state.learning_state.learning_memory.current_session_log = []
                                        st.session_state.learning_state.learning_memory.current_session_log.append(log_entry)
                                        
                                        st.write(f"⬇️ Installing {package}...")
                                        subprocess.check_call([
                                            sys.executable, "-m", "pip", "install", "-q", package
                                        ])
                                        st.write(f"✅ {package} installed successfully")
                        
                        # Create namespace and execute in correct order
                        globals_dict = {"__builtins__": __builtins__}
                        
                        # 1. Execute all imports first
                        if import_lines:
                            exec('\n'.join(import_lines), globals_dict)
                        
                        # 2. Execute the rest of the code with output capture
                        with contextlib.redirect_stdout(output_buffer):
                            exec('\n'.join(code_lines), globals_dict)
                        
                        user_output = output_buffer.getvalue()
                        st.success("✅ Execution Successful!")
                        st.code(user_output or "No output printed.")

                        # Log interaction for assessment
                        log_entry = f"Exercise {idx} ('{ex.title}') execution: SUCCESS. Output excerpt: {user_output[:100]}..."
                        if not hasattr(st.session_state.learning_state.learning_memory, 'current_session_log'):
                             st.session_state.learning_state.learning_memory.current_session_log = []
                        st.session_state.learning_state.learning_memory.current_session_log.append(log_entry)
                        
                        # Automated Evaluation
                        st.markdown("#### ⚖️ Evaluation & Feedback")
                        with st.status("Evaluating your code..."):
                            eval_prompt = f"""
                            You are an expert Python evaluator. Compare the User's Code and Output with the Reference Solution.
                            Assign a score (0-100) and provide specific feedback for improvement.
                            
                            Exercise: {ex.title}
                            Description: {ex.description}
                            
                            Reference Solution:
                            {ex.solution}
                            
                            User Code:
                            {code_input}
                            
                            User Output:
                            {user_output}
                            
                            Return the result in JSON format:
                            {{
                                "score": int,
                                "feedback": "concise feedback",
                                "suggestions": ["suggestion 1", "suggestion 2"]
                            }}
                            """
                            from workflow import llm
                            eval_response = llm.invoke(eval_prompt).content
                            try:
                                eval_data = json.loads(eval_response.strip("`json\n"))
                                st.metric("Evaluation Score", f"{eval_data['score']}/100")
                                st.info(eval_data['feedback'])
                                if eval_data['suggestions']:
                                    st.markdown("**Suggestions for improvement:**")
                                    for sugg in eval_data['suggestions']:
                                        st.write(f"- {sugg}")
                            except Exception:
                                st.warning("Could not parse evaluation results, but well done on running the code!")

                    except Exception as e:
                        error_msg = traceback.format_exc()
                        st.error("❌ Execution Failed!")
                        st.code(error_msg)
                        
                        # Store error in memory for adaptive feedback
                        learning_state.learning_memory.notes.append(f"Exercise {idx} ('{ex.title}') failed with error: {str(e)}")
                        st.info("The agent has noted this struggle and will adapt future lessons.")
                        
                        # Show solution
                        with st.expander("💡 View Masterclass Solution", expanded=False):
                            st.code(ex.solution)

                st.divider()
    else:
        st.info("No exercises available.")

# Assess learning
if st.button("📝 Assess Learning"):
    assessment = assess_learning(learning_state)
    learning_state.learning_assessment = assessment

# Persistent Display: Assessment
if learning_state.learning_assessment and learning_state.learning_assessment.score is not None:
    st.divider()
    st.markdown("### 📊 Assessment Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score", f"{learning_state.learning_assessment.score * 100:.0f}%")
    with col2:
        st.metric("Confidence", f"{learning_state.learning_assessment.confidence * 100:.0f}%")
    with col3:
        st.metric("Error Type", learning_state.learning_assessment.error_type.title())

    if learning_state.learning_assessment.analysis:
        st.info(f"📋 **Personalized Feedback:** {learning_state.learning_assessment.analysis}")
    else:
        st.info(f"📋 **Feedback:** {learning_state.learning_assessment.feedback}")

# Evaluate next steps
if st.button("⚡ Evaluate Next Step"):
    with st.spinner("Evaluating next step..."):
        result = asyncio.run(evaluate(learning_state))
        learning_state.learning_assessment = result["learning_assessment"]

# Persistent Display: Next Step Evaluation
if learning_state.learning_assessment and learning_state.learning_assessment.next_steps:
    next_step = learning_state.learning_assessment.next_steps
    # Use next_step_analysis if available, otherwise fallback
    analysis_text = getattr(learning_state.learning_assessment, 'next_step_analysis', None) or learning_state.learning_assessment.analysis
    
    if next_step != "None":
        st.divider()
        st.success(f"Recommended Next Step: **{next_step.replace('_', ' ').title()}**")
        if analysis_text:
            st.info(f"🚀 **Strategy Analysis:** {analysis_text}")

# Execute next steps adaptively
next_step = learning_state.learning_assessment.next_steps

# Display persistent success/result message
if "execution_result" in st.session_state and st.session_state.execution_result:
    st.divider()
    st.markdown(f"### ✨ Result: {st.session_state.get('execution_step_name', 'Action')}")
    st.info(st.session_state.execution_result)
    
    if st.button("Clear Result", key="clear_res"):
        del st.session_state.execution_result
        st.rerun()
elif "success_msg" in st.session_state and st.session_state.success_msg:
    st.success(st.session_state.success_msg)
    del st.session_state.success_msg

if next_step and next_step != "None":
    st.markdown(f"### 🔄 Execute Next Step")
    
    if st.button(f"🚀 Start {next_step.replace('_', ' ').title()}"):
        with st.spinner(f"Applying {next_step}..."):
            if next_step == "reteach_theory":
                result = asyncio.run(reteach_theory(learning_state))
            elif next_step == "reteach_examples":
                result = asyncio.run(reteach_examples(learning_state))
            elif next_step == "reinforce":
                result = asyncio.run(reinforce(learning_state))
            elif next_step == "advance":
                result = asyncio.run(advance(learning_state))
            else:
                st.warning("Unknown next step!")
                result = None
            
            if result:
                # Store detailed result for display
                if isinstance(result, dict) and "execution_result" in result:
                     st.session_state.execution_result = result["execution_result"]
                     st.session_state.execution_step_name = next_step.replace('_', ' ').title()
                else:
                     st.session_state.success_msg = f"Step '{next_step}' executed successfully! Content updated."

                # Reset next_steps after execution
                learning_state.learning_assessment.next_steps = "None"
                st.rerun()
    st.markdown("### Learning Memory")
    st.json(learning_state.learning_memory.model_dump())

# Debugging / Session State
with st.expander("📂 View Full Session State"):
    st.json(learning_state.model_dump())