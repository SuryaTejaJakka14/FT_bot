# src/modules/skills_taxonomy.py
"""
Skills Taxonomy Module
Comprehensive database of technical and soft skills for resume parsing.
"""

from typing import List, Dict, Set
import re


class SkillsTaxonomy:
    """
    Comprehensive taxonomy of skills with normalization and extraction methods.
    
    Categories:
    - Programming Languages
    - ML/AI Technologies
    - Databases
    - Cloud/DevOps
    - Web Frameworks
    - Data Tools
    - IoT/Embedded
    - Other Technical
    - Soft Skills
    """
    
    def __init__(self):
        """Initialize the taxonomy with all skill categories."""
        self._build_taxonomy()
        self._build_aliases()
    
    def _build_taxonomy(self):
        """Build the complete skills taxonomy."""
        # We'll add this next
        pass
    
    def _build_aliases(self):
        """
        Build skill aliases for normalization.
        Maps variations to canonical form.

        Example: "py" → "python", "k8s" → "kubernetes"
        """
        self.aliases = {
            # Programming language aliases
            "py": "python",
            "python3": "python",
            "js": "javascript",
            "ts": "typescript",
            "golang": "go",
            
            # ML/AI aliases
            "tf": "tensorflow",
            "sklearn": "scikit-learn",
            "cv": "computer vision",
            "dl": "deep learning",
            "ml": "machine learning",
            
            # Cloud/DevOps aliases
            "k8s": "kubernetes",
            "kube": "kubernetes",
            "aws ec2": "ec2",
            "aws s3": "s3",
            "aws lambda": "lambda",
            
            # Database aliases
            "postgres": "postgresql",
            "psql": "postgresql",
            "mongo": "mongodb",
            
            # Web framework aliases
            "reactjs": "react",
            "vuejs": "vue",
            "nodejs": "node",
            "node.js": "node",
            "next.js": "nextjs",
            
            # IoT/Embedded aliases
            "raspi": "raspberry pi",
            "rpi": "raspberry pi",
            "embedded-c": "embedded c",
            
            # Other aliases
            "github actions": "github actions",
            "ci-cd": "ci/cd",
            "restful api": "rest api",
            "rest-api": "rest api"
        }



    def _build_taxonomy(self):
        """Build the complete skills taxonomy."""
        
        # ========== PROGRAMMING LANGUAGES ==========
        self.programming_languages = {
            "python", "java", "javascript", "typescript", "c++", "c", "c#",
            "go", "golang", "rust", "ruby", "php", "swift", "kotlin",
            "scala", "r", "matlab", "perl", "shell", "bash", "powershell",
            "objective-c", "dart", "lua", "haskell", "elixir"
        }
        
        # ========== ML/AI TECHNOLOGIES ==========
        self.ml_ai = {
            "machine learning", "deep learning", "neural networks",
            "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
            "pandas", "numpy", "opencv", "nltk", "spacy", "transformers",
            "hugging face", "bert", "gpt", "llm", "large language models",
            "computer vision", "nlp", "natural language processing",
            "reinforcement learning", "supervised learning",
            "unsupervised learning", "gradient descent", "backpropagation",
            "cnn", "rnn", "lstm", "gan", "autoencoder"
        }
        
        # ========== DATABASES ==========
        self.databases = {
            "sql", "mysql", "postgresql", "postgres", "sqlite", "mongodb",
            "redis", "cassandra", "dynamodb", "neo4j", "elasticsearch",
            "oracle", "sql server", "mariadb", "couchdb", "influxdb",
            "timescaledb", "clickhouse", "bigquery", "snowflake",
            "database design", "data modeling", "orm", "sqlalchemy"
        }
        
        # ========== CLOUD & DEVOPS ==========
        self.cloud_devops = {
            "aws", "amazon web services", "azure", "gcp", "google cloud",
            "docker", "kubernetes", "k8s", "terraform", "ansible",
            "jenkins", "gitlab ci", "github actions", "circleci",
            "ci/cd", "devops", "infrastructure as code", "iac",
            "ec2", "s3", "lambda", "ecs", "eks", "cloudformation",
            "nginx", "apache", "load balancing", "microservices",
            "serverless", "api gateway", "cloudwatch", "prometheus",
            "grafana", "elk stack", "datadog", "new relic"
        }
        
        # ========== WEB FRAMEWORKS & TECHNOLOGIES ==========
        self.web_frameworks = {
            "react", "reactjs", "angular", "vue", "vuejs", "nextjs",
            "node", "nodejs", "express", "expressjs", "django",
            "flask", "fastapi", "spring", "spring boot", "asp.net",
            "rails", "ruby on rails", "laravel", "symfony",
            "html", "html5", "css", "css3", "sass", "scss", "less",
            "tailwind", "bootstrap", "jquery", "ajax", "rest api",
            "restful", "graphql", "websocket", "http", "https",
            "oauth", "jwt", "authentication", "authorization"
        }
        
        # ========== DATA TOOLS & ANALYTICS ==========
        self.data_tools = {
            "spark", "apache spark", "hadoop", "hive", "pig", "kafka",
            "airflow", "luigi", "dask", "ray", "tableau", "power bi",
            "looker", "qlik", "excel", "jupyter", "colab",
            "data warehousing", "etl", "data pipeline", "data engineering",
            "data analysis", "data visualization", "statistics",
            "a/b testing", "experimentation", "analytics"
        }
        
        # ========== IOT & EMBEDDED SYSTEMS ==========
        self.iot_embedded = {
            "embedded c", "embedded systems", "rtos", "freertos",
            "arduino", "raspberry pi", "arm", "arm cortex",
            "microcontroller", "mcu", "fpga", "verilog", "vhdl",
            "i2c", "spi", "uart", "can bus", "modbus", "mqtt",
            "iot", "internet of things", "sensor integration",
            "firmware", "bootloader", "device driver", "kernel"
        }
        
        # ========== OTHER TECHNICAL SKILLS ==========
        self.other_technical = {
            "git", "github", "gitlab", "bitbucket", "version control",
            "agile", "scrum", "kanban", "jira", "confluence",
            "linux", "unix", "windows", "macos", "vim", "emacs",
            "vscode", "intellij", "eclipse", "xcode",
            "testing", "unit testing", "integration testing", "pytest",
            "jest", "selenium", "cypress", "test automation",
            "debugging", "profiling", "optimization",
            "algorithms", "data structures", "design patterns",
            "object-oriented programming", "oop", "functional programming",
            "multithreading", "concurrency", "parallel processing",
            "networking", "tcp/ip", "dns", "vpn", "security",
            "encryption", "cryptography", "penetration testing"
        }
        
        # ========== COMBINE ALL HARD SKILLS ==========
        self.all_hard_skills = (
            self.programming_languages |
            self.ml_ai |
            self.databases |
            self.cloud_devops |
            self.web_frameworks |
            self.data_tools |
            self.iot_embedded |
            self.other_technical
        )
        # ========== SOFT SKILLS ==========
        self.soft_skills = {
            "leadership", "team leadership", "project management",
            "communication", "verbal communication", "written communication",
            "teamwork", "collaboration", "cross-functional collaboration",
            "problem solving", "analytical thinking", "critical thinking",
            "creativity", "innovation", "adaptability", "flexibility",
            "time management", "organization", "attention to detail",
            "mentoring", "coaching", "stakeholder management",
            "presentation", "public speaking", "negotiation",
            "conflict resolution", "decision making", "strategic thinking",
            "emotional intelligence", "empathy", "customer service",
            "business acumen", "product management", "user research",
            "documentation", "technical writing", "requirement gathering"
        }

    def normalize_skill(self, skill: str) -> str:
        """
        Normalize a skill name to its canonical form.
        
        Steps:
        1. Convert to lowercase
        2. Strip whitespace
        3. Check aliases
        4. Return normalized form
        
        Args:
            skill: Raw skill name from resume
            
        Returns:
            Normalized skill name
            
        Examples:
            >>> normalize_skill("Python")
            "python"
            >>> normalize_skill("K8S")
            "kubernetes"
            >>> normalize_skill("  TensorFlow  ")
            "tensorflow"
        """
        # Step 1: Lowercase and strip
        normalized = skill.lower().strip()
        
        # Step 2: Check aliases
        normalized = self.aliases.get(normalized, normalized)
        
        # Step 3: Handle special cases
        # Remove version numbers: "python 3.11" → "python"
        normalized = re.sub(r'\s+\d+(\.\d+)*$', '', normalized)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized


    def extract_skills_from_text(self, text: str) -> tuple[List[str], List[str]]:
        """
        Extract both hard and soft skills from text.
        
        Algorithm:
        1. Normalize text (lowercase)
        2. For each skill in taxonomy:
           - Check if skill appears in text (word boundary matching)
           - Normalize found skill
           - Add to results
        3. Deduplicate and sort
        
        Args:
            text: Resume text to extract skills from
            
        Returns:
            Tuple of (hard_skills, soft_skills)
            
        Example:
            >>> text = "I'm experienced with Python, AWS, and leadership"
            >>> hard, soft = taxonomy.extract_skills_from_text(text)
            >>> print(hard)
            ["aws", "python"]
            >>> print(soft)
            ["leadership"]
        """
        text_lower = text.lower()
        found_hard_skills = set()
        found_soft_skills = set()
        
        # ========== EXTRACT HARD SKILLS ==========
        for skill in self.all_hard_skills:
            # Use word boundary matching to avoid partial matches
            # Example: "java" matches "java" but not "javascript"
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                normalized = self.normalize_skill(skill)
                found_hard_skills.add(normalized)
        
        # ========== EXTRACT SOFT SKILLS ==========
        for skill in self.soft_skills:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                normalized = self.normalize_skill(skill)
                found_soft_skills.add(normalized)
        
        # ========== RETURN SORTED LISTS ==========
        return (
            sorted(list(found_hard_skills)),
            sorted(list(found_soft_skills))
        )


    def get_skill_category(self, skill: str) -> str:
        """
        Get the category of a given skill.
        
        Args:
            skill: Normalized skill name
            
        Returns:
            Category name or "unknown"
            
        Example:
            >>> get_skill_category("python")
            "programming_languages"
            >>> get_skill_category("tensorflow")
            "ml_ai"
        """
        normalized = self.normalize_skill(skill)
        
        if normalized in self.programming_languages:
            return "programming_languages"
        elif normalized in self.ml_ai:
            return "ml_ai"
        elif normalized in self.databases:
            return "databases"
        elif normalized in self.cloud_devops:
            return "cloud_devops"
        elif normalized in self.web_frameworks:
            return "web_frameworks"
        elif normalized in self.data_tools:
            return "data_tools"
        elif normalized in self.iot_embedded:
            return "iot_embedded"
        elif normalized in self.other_technical:
            return "other_technical"
        elif normalized in self.soft_skills:
            return "soft_skills"
        else:
            return "unknown"


    def get_statistics(self) -> Dict[str, int]:
        """
        Get taxonomy statistics.
        
        Returns:
            Dictionary with counts per category
        """
        return {
            "programming_languages": len(self.programming_languages),
            "ml_ai": len(self.ml_ai),
            "databases": len(self.databases),
            "cloud_devops": len(self.cloud_devops),
            "web_frameworks": len(self.web_frameworks),
            "data_tools": len(self.data_tools),
            "iot_embedded": len(self.iot_embedded),
            "other_technical": len(self.other_technical),
            "soft_skills": len(self.soft_skills),
            "total_hard_skills": len(self.all_hard_skills),
            "total_skills": len(self.all_hard_skills) + len(self.soft_skills),
            "aliases": len(self.aliases)
        }
