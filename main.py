from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
import json
from datetime import datetime
import hashlib
import asyncio
import logging
import os

# Optional imports with proper error handling
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not installed. GPT-4 features will be disabled.")

try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: torch/transformers not installed. Local LLM features will be disabled.")

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("Warning: aiohttp not installed. Some async features may be limited.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CREATE APP FIRST
app = FastAPI(
    title="Real AI Code Review Assistant",
    description="🚀 True AI-powered code analysis with real-time LLM suggestions",
    version="2.0.0"
)

# ADD MIDDLEWARE IMMEDIATELY AFTER APP CREATION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - SECURITY WARNING: Don't hardcode API keys!
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

if OPENAI_API_KEY and OPENAI_AVAILABLE:
    openai.api_key = OPENAI_API_KEY

# DEFINE MODELS
class CodeReviewRequest(BaseModel):
    code: str
    language: str = "python"
    context: Optional[str] = None
    ai_model: str = "rule_based"

class AICodeIssue(BaseModel):
    line: int
    severity: str
    category: str
    message: str
    ai_suggestion: str
    ai_explanation: str
    fixed_code: Optional[str] = None
    confidence: float

class AIReviewResponse(BaseModel):
    review_id: str
    timestamp: str
    overall_score: float
    issues: List[AICodeIssue]
    ai_summary: str
    ai_improvements: List[str]
    refactored_code: Optional[str] = None
    code_quality_metrics: Dict[str, Any] 

# SERVE HTML FILE AT ROOT
@app.get("/")
async def root():
    return FileResponse("index.html")

@app.get("/ui")
async def get_ui():
    return FileResponse("index.html")

# Global variable for caching LLM
code_llm = None

def get_code_llm():
    """Initialize local code LLM with proper error handling"""
    global code_llm
    if not TRANSFORMERS_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Transformers library not available. Install with: pip install torch transformers"
        )
    
    if code_llm is None:
        try:
            code_llm = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            logger.info("Local LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize local LLM: {e}")
            raise HTTPException(status_code=503, detail=f"LLM initialization failed: {str(e)}")
    
    return code_llm

async def analyze_with_gpt4(code: str, language: str) -> Dict:
    """Use GPT-4 for real-time code analysis with proper error handling"""
    
    if not OPENAI_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="OpenAI package not available. Install with: pip install openai"
        )
    
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=401, 
            detail="OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
        )
    
    system_prompt = f"""You are an expert {language} code reviewer. Analyze the provided code and return a JSON response with:
1. Issues found (line number, severity, category, detailed explanation, specific suggestion)
2. Overall quality score (0-100)
3. Detailed improvement recommendations
4. Refactored version of the code if improvements are needed

Focus on:
- Security vulnerabilities
- Performance optimizations  
- Code maintainability
- Best practices
- Potential bugs

Be specific and actionable in your suggestions."""

    user_prompt = f"""Please review this {language} code:

```{language}
{code}
```

Return response as JSON with this structure:
{{
    "overall_score": 85,
    "issues": [
        {{
            "line": 5,
            "severity": "high",
            "category": "security",
            "message": "Specific issue description",
            "suggestion": "Specific fix recommendation",
            "explanation": "Why this is problematic",
            "confidence": 0.95
        }}
    ],
    "summary": "AI-generated summary of code quality",
    "improvements": ["specific improvement 1", "specific improvement 2"],
    "refactored_code": "improved version if needed"
}}"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        ai_response = response.choices[0].message.content
        return json.loads(ai_response)
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse GPT-4 JSON response: {e}")
        return {"error": "Invalid JSON response from GPT-4", "raw_response": ai_response}
    except Exception as e:
        logger.error(f"GPT-4 analysis failed: {e}")
        return {"error": str(e)}

def get_suggestion(pattern: str, language: str) -> str:
    """Get specific suggestions for detected patterns"""
    suggestions = {
        'eval(': 'Use ast.literal_eval() for safe evaluation or validate input properly',
        'exec(': 'Avoid exec() or use safer alternatives like importlib for dynamic imports',
        'subprocess.call': 'Use subprocess.run() with shell=False and validate inputs',
        'pickle.loads': 'Use json for data serialization or validate pickle sources',
        'yaml.load(': 'Replace with yaml.safe_load() to prevent code execution',
        'for i in range(len(': 'Use "for i, item in enumerate(items):" instead',
        '+ str(': 'Use f-strings: f"text{variable}" or str.join() for multiple concatenations',
        'except:': 'Specify exception types: "except ValueError:" or "except (TypeError, ValueError):"',
        'import *': 'Import specific functions: "from module import function1, function2"',
        'global ': 'Pass variables as function parameters or use class attributes',
        'print(': 'Use logging: "import logging; logging.info(message)"',
        'var ': 'Use "let" for variables that change or "const" for constants',
        '== ': 'Use === for strict comparison that checks type and value',
        'innerHTML': 'Use textContent for text or sanitize HTML content',
        'document.write': 'Use safer DOM manipulation methods'
    }
    
    return suggestions.get(pattern, f"Review usage of '{pattern}' for best practices")
# Place this somewhere above analyze_with_rule_based()
import ast

def analyze_ast(code):
    issues = []
    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, 'id', '') in ['eval', 'exec']:
                issues.append({
                    "line": node.lineno,
                    "severity": "high",
                    "category": "security",
                    "message": f"Use of `{node.func.id}` is dangerous.",
                    "suggestion": f"Avoid using `{node.func.id}`; consider safer alternatives.",
                    "explanation": f"`{node.func.id}` can execute arbitrary code.",
                    "confidence": 0.95
                })

            elif isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append({
                    "line": node.lineno,
                    "severity": "medium",
                    "category": "error-handling",
                    "message": "Bare except detected.",
                    "suggestion": "Catch specific exceptions instead of using bare except.",
                    "explanation": "Bare except can mask unexpected errors.",
                    "confidence": 0.9
                })

            elif isinstance(node, ast.FunctionDef) and len(node.args.args) > 5:
                issues.append({
                    "line": node.lineno,
                    "severity": "low",
                    "category": "design",
                    "message": "Function has too many arguments.",
                    "suggestion": "Consider refactoring to use fewer arguments.",
                    "explanation": "Functions with many arguments are harder to test and maintain.",
                    "confidence": 0.8
                })

    except SyntaxError as e:
        issues.append({
            "line": e.lineno or 1,
            "severity": "high",
            "category": "syntax",
            "message": "Syntax error in code",
            "suggestion": "Fix the syntax error",
            "explanation": str(e),
            "confidence": 0.99
        })

    return issues


async def analyze_with_rule_based(code: str, language: str) -> Dict:
    """Enhanced rule-based analysis that properly analyzes code structure"""
    
    issues = []
    
    # 🔍 AST-based security checks for Python
    if language.lower() == "python":
        issues += analyze_ast(code)
    
    score = 100
    lines = code.split('\n')
    
    # Language-specific patterns
    if language.lower() == "python":
        patterns = {
            # Security issues (high severity)
            'eval(': ('security', 'high', 'Dangerous eval() usage - can execute arbitrary code'),
            'exec(': ('security', 'high', 'Dangerous exec() usage - can execute arbitrary code'),
            'subprocess.call': ('security', 'medium', 'Subprocess call should use shell=False for security'),
            'pickle.loads': ('security', 'high', 'Unsafe pickle deserialization can execute arbitrary code'),
            'yaml.load(': ('security', 'high', 'Unsafe YAML loading - use yaml.safe_load() instead'),
            'input(': ('security', 'medium', 'Raw input() can be dangerous - validate user input'),
            
            # Performance issues (medium severity)
            'for i in range(len(': ('performance', 'medium', 'Use enumerate() instead of range(len()) for better readability'),
            '+ str(': ('performance', 'medium', 'String concatenation in loop - consider f-strings or join()'),
            '== True': ('performance', 'low', 'Redundant comparison with True - use "if variable:" instead'),
            '== False': ('performance', 'low', 'Redundant comparison with False - use "if not variable:" instead'),
            
            # Error handling (medium severity)
            'except:': ('error_handling', 'medium', 'Bare except clause catches all exceptions - specify exception types'),
            'pass': ('error_handling', 'low', 'Empty pass statement - consider logging or proper error handling'),
            
            # Style issues (low severity)
            'import *': ('style', 'medium', 'Avoid wildcard imports - import specific functions/classes'),
            'global ': ('style', 'medium', 'Global variables should be avoided - use function parameters'),
            'print(': ('style', 'low', 'Consider using logging instead of print for production code'),
        }
    elif language.lower() in ["javascript", "js"]:
        patterns = {
            # Security issues
            'eval(': ('security', 'high', 'eval() is dangerous and can execute arbitrary code'),
            'innerHTML': ('security', 'medium', 'innerHTML can lead to XSS attacks - use textContent or sanitize'),
            'document.write': ('security', 'medium', 'document.write can be exploited for XSS attacks'),
            
            # Performance issues
            'var ': ('performance', 'low', 'Use let or const instead of var for better scoping'),
            '== ': ('performance', 'low', 'Use === for strict equality comparison'),
            '!= ': ('performance', 'low', 'Use !== for strict inequality comparison'),
            
            # Style issues
            'function(': ('style', 'low', 'Consider using arrow functions for shorter syntax'),
        }
    elif language.lower() in ["java"]:
        patterns = {
            'String +': ('performance', 'medium', 'Use StringBuilder for multiple string concatenations'),
            'catch (Exception': ('error_handling', 'medium', 'Catch specific exceptions instead of generic Exception'),
            'System.out.print': ('style', 'low', 'Use logging framework instead of System.out'),
        }
    elif language.lower() in ["cpp", "c++"]:
        patterns = {
            'malloc(': ('memory', 'medium', 'Consider using smart pointers instead of raw malloc'),
            'gets(': ('security', 'high', 'gets() is unsafe - use fgets() instead'),
            'strcpy(': ('security', 'medium', 'strcpy can cause buffer overflow - use strncpy'),
        }
    else:
        # Generic patterns for other languages
        patterns = {
            'TODO': ('style', 'low', 'TODO comment found - consider completing or removing'),
            'FIXME': ('style', 'medium', 'FIXME comment found - indicates a known issue'),
            'XXX': ('style', 'medium', 'XXX comment found - indicates problematic code'),
        }
    
    # Analyze each line
    for line_num, line in enumerate(lines, 1):
        line_stripped = line.strip()
        
        # Skip empty lines and comments
        if not line_stripped or line_stripped.startswith('#') or line_stripped.startswith('//'):
            continue
            
        # Check for patterns
        for pattern, (category, severity, message) in patterns.items():
            if pattern.lower() in line.lower():
                confidence = 0.9 if severity == 'high' else 0.8 if severity == 'medium' else 0.7
                
                issues.append({
                    "line": line_num,
                    "severity": severity,
                    "category": category,
                    "message": message,
                    "suggestion": get_suggestion(pattern, language),
                    "explanation": f"Found '{pattern}' on line {line_num}: {line_stripped[:50]}{'...' if len(line_stripped) > 50 else ''}",
                    "confidence": confidence
                })
                
                # Deduct points based on severity
                score -= 15 if severity == 'high' else 8 if severity == 'medium' else 3
    
    # Additional code quality checks
    total_lines = len([line for line in lines if line.strip()])
    comment_lines = len([line for line in lines if line.strip().startswith('#') or line.strip().startswith('//')])
    comment_ratio = comment_lines / max(total_lines, 1)
    
    # Check for very long lines (Python PEP 8: 79 chars, general: 120 chars)
    max_line_length = 79 if language.lower() == "python" else 120
    long_lines = [i+1 for i, line in enumerate(lines) if len(line) > max_line_length]
    
    for line_num in long_lines:
        issues.append({
            "line": line_num,
            "severity": "low",
            "category": "style",
            "message": f"Line exceeds {max_line_length} characters",
            "suggestion": "Break long lines into multiple lines for better readability",
            "explanation": f"Line {line_num} is {len(lines[line_num-1])} characters long",
            "confidence": 0.9
        })
        score -= 2
    
    # Check comment ratio
    if comment_ratio < 0.1 and total_lines > 10:
        issues.append({
            "line": 1,
            "severity": "low",
            "category": "documentation",
            "message": "Low comment ratio - consider adding more documentation",
            "suggestion": "Add comments to explain complex logic and function purposes",
            "explanation": f"Only {comment_ratio:.1%} of lines are comments",
            "confidence": 0.7
        })
        score -= 5
    
    # Function/method detection (basic)
    function_count = 0
    if language.lower() == "python":
        function_count = len([line for line in lines if line.strip().startswith('def ')])
    elif language.lower() in ["javascript", "js"]:
        function_count = len([line for line in lines if 'function' in line or '=>' in line])
    elif language.lower() == "java":
        function_count = len([line for line in lines if 'public ' in line and '(' in line])
    
    # Generate improvements based on found issues
    improvements = []
    categories = set(issue["category"] for issue in issues)
    
    if "security" in categories:
        improvements.append("Review security practices - avoid dangerous functions like eval()")
    if "performance" in categories:
        improvements.append("Optimize performance - use efficient loops and comparisons")
    if "error_handling" in categories:
        improvements.append("Improve error handling with specific exception types")
    if "style" in categories:
        improvements.append("Follow coding style guidelines for better maintainability")
    if "memory" in categories:
        improvements.append("Review memory management and consider safer alternatives")
    if comment_ratio < 0.15:
        improvements.append("Add more comments and documentation")
    
    if not improvements:
        improvements = [
            "Code looks good! Consider adding unit tests",
            "Consider adding type hints for better code documentation",
            "Ensure proper error handling for edge cases"
        ]
    
    # Calculate final score
    final_score = max(score, 0)
    
    # Generate summary based on results
    if final_score >= 90:
        summary = f"Excellent code quality! Found {len(issues)} minor suggestions for improvement."
    elif final_score >= 75:
        summary = f"Good code quality with {len(issues)} issues to address for better maintainability."
    elif final_score >= 60:
        summary = f"Moderate code quality - {len(issues)} issues found that should be addressed."
    else:
        summary = f"Code needs improvement - {len(issues)} issues found including security and performance concerns."
    
    return {
        "overall_score": final_score,
        "issues": issues,
        "summary": summary,
        "improvements": improvements,
        "code_quality_metrics": {
            "lines_of_code": total_lines,
            "comment_ratio": comment_ratio,
            "function_count": function_count,
            "cyclomatic_complexity": min(len(issues) + 1, 10)  # Simplified complexity
        }
    }

async def analyze_with_local_llm(code: str, language: str) -> Dict:
    """Use local LLM for analysis with better error handling and fallback"""
    
    try:
        llm = get_code_llm()
        
        # Create a more specific prompt
        prompt = f"""Review this {language} code and identify issues:

```{language}
{code[:800]}  # Limit code length for model
```

Find problems with:
1. Security vulnerabilities
2. Performance issues  
3. Code style problems
4. Potential bugs

Provide specific suggestions."""

        # Generate response
        response = llm(prompt, max_length=300, temperature=0.1, do_sample=True, pad_token_id=llm.tokenizer.eos_token_id)
        ai_text = response[0]['generated_text']
        
        # Extract the generated part (remove the original prompt)
        if prompt in ai_text:
            ai_text = ai_text.replace(prompt, "").strip()
        
        # Parse the AI response to extract issues (basic parsing)
        issues = []
        if ai_text:
            # Look for common issue indicators in the response
            lines = ai_text.split('\n')
            for i, line in enumerate(lines[:5]):  # Limit to first 5 lines
                if any(word in line.lower() for word in ['issue', 'problem', 'error', 'warning', 'improve']):
                    issues.append({
                        "line": i + 1,
                        "severity": "medium",
                        "category": "ai_analysis",
                        "message": f"AI detected potential issue: {line[:100]}",
                        "suggestion": line[:150] if line else "Consider reviewing this section",
                        "explanation": "Analysis from local language model",
                        "confidence": 0.6
                    })
        
        # If no specific issues found, add a general analysis
        if not issues:
            issues.append({
                "line": 1,
                "severity": "low",
                "category": "general",
                "message": "Local LLM analysis completed",
                "suggestion": ai_text[:200] if ai_text else "Code structure appears reasonable",
                "explanation": "General analysis from local language model",
                "confidence": 0.5
            })
        
        score = max(85 - len(issues) * 5, 50)  # Base score with deductions
        
        return {
            "overall_score": score,
            "issues": issues,
            "summary": f"Local LLM analysis found {len(issues)} areas for potential improvement",
            "improvements": [
                "Consider the AI suggestions provided",
                "Review code for common patterns and best practices",
                "Add appropriate error handling and validation"
            ],
            "code_quality_metrics": {
                "lines_of_code": len(code.split('\n')),
                "comment_ratio": 0.1,
                "function_count": code.count('def ') + code.count('function'),
                "cyclomatic_complexity": min(len(issues) + 1, 8) 
            },
            "model_used": "Local LLM"
        }
        
    except Exception as e:
        logger.error(f"Local LLM analysis failed: {e}")
        # Fallback to rule-based analysis
        return await analyze_with_rule_based(code, language)

@app.post("/ai-review", response_model=AIReviewResponse)
async def ai_code_review(request: CodeReviewRequest):
    """🤖 Get AI-powered code review with proper fallbacks"""
    
    try:
        review_id = hashlib.md5(f"{request.code}{datetime.now()}".encode()).hexdigest()[:8]
        
        # Choose analysis method based on availability and request
        if request.ai_model == "gpt-4" and OPENAI_AVAILABLE and OPENAI_API_KEY:
            ai_result = await analyze_with_gpt4(request.code, request.language)
        elif request.ai_model == "local" and TRANSFORMERS_AVAILABLE:
            ai_result = await analyze_with_local_llm(request.code, request.language)
        else:
            ai_result = await analyze_with_rule_based(request.code, request.language)
        
        if "error" in ai_result:
            logger.warning(f"AI analysis failed: {ai_result['error']}")
            ai_result = await analyze_with_rule_based(request.code, request.language)
        
        # Convert AI response to our format
        ai_issues = []
        for issue_data in ai_result.get("issues", []):
            ai_issues.append(AICodeIssue(
                line=issue_data.get("line", 1),
                severity=issue_data.get("severity", "medium"),
                category=issue_data.get("category", "general"),
                message=issue_data.get("message", "Issue detected"),
                ai_suggestion=issue_data.get("suggestion", "No specific suggestion"),
                ai_explanation=issue_data.get("explanation", "Analysis completed"),
                fixed_code=issue_data.get("fixed_code"),
                confidence=issue_data.get("confidence", 0.7)
            ))
        
        # Add default metrics if not provided
        if 'code_quality_metrics' not in ai_result:
            ai_result['code_quality_metrics'] = {
                'lines_of_code': len(request.code.split('\n')),
                'cyclomatic_complexity': 1,  # Default simple complexity
                'function_count': request.code.count('def '),
                'comment_ratio': 0.1  # Default 10% comments
            }
        
        return AIReviewResponse(
            review_id=review_id,
            timestamp=datetime.now().isoformat(),
            overall_score=ai_result.get("overall_score", 75),
            issues=ai_issues,
            ai_summary=ai_result.get("summary", "Analysis completed"),
            ai_improvements=ai_result.get("improvements", []),
            refactored_code=ai_result.get("refactored_code"),
            code_quality_metrics=ai_result['code_quality_metrics']
        )
        
    except Exception as e:
        logger.error(f"Review failed: {e}")
        raise HTTPException(status_code=500, detail=f"Review failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint with service status"""
    return {
        "status": "healthy",
        "services": {
            "openai": OPENAI_AVAILABLE and bool(OPENAI_API_KEY),
            "transformers": TRANSFORMERS_AVAILABLE,
            "aiohttp": AIOHTTP_AVAILABLE
        },
        "available_models": get_available_models()
    }

def get_available_models():
    """Get list of available AI models based on installed packages"""
    models = ["rule_based"]
    
    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        models.append("gpt-4")
    
    if TRANSFORMERS_AVAILABLE:
        models.append("local")
    
    return models

@app.get("/ai-models")
async def available_models():
    """List available AI models for code review"""
    base_models = [
        {
            "name": "rule_based",
            "description": "Fast rule-based analysis - always available",
            "speed": "very_fast",
            "cost": "free",
            "accuracy": "good",
            "available": True
        }
    ]
    
    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        base_models.append({
            "name": "gpt-4",
            "description": "OpenAI GPT-4 - Highest quality analysis",
            "speed": "slow",
            "cost": "high",
            "accuracy": "excellent",
            "available": True
        })
    
    if TRANSFORMERS_AVAILABLE:
        base_models.append({
            "name": "local",
            "description": "Local LLM - Good balance of speed and accuracy",
            "speed": "medium",
            "cost": "free",
            "accuracy": "good",
            "available": True
        })
    
    return {"models": base_models}

if __name__ == "__main__":
    print("🤖 Starting AI Code Review Assistant...")
    print(f"📊 Available services: OpenAI={OPENAI_AVAILABLE}, Transformers={TRANSFORMERS_AVAILABLE}")
    print(f"🔑 API Keys configured: OpenAI={bool(OPENAI_API_KEY)}")
    print("🚀 Server starting...")
    
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)