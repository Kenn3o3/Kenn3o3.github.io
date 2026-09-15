---
permalink: /
title: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I am Kenny Wong Lik Hang (王力恒). I received my **BSc in Computer Science** from **City University of Hong Kong (CityUHK)** in 2025 and my **MSc in Computer Science and Engineering** from **The Chinese University of Hong Kong (CUHK)** in 2026.

I am an **MPhil student in Computer Science and Engineering** at **CUHK** (from **1 August 2026 to 31 July 2028**), under the supervision of [**Prof. Dou Qi**](https://www.cse.cuhk.edu.hk/~qdou/). My research interest is in Embodied AI and Robotics.

You can view my CV ([here](/files/resume.pdf)). Feel free to reach out to me at `klhwong3 [at] outlook [dot] com`—I’d love to collaborate on something exciting in the future! :>

<a href="https://visitorbadge.io/status?path=https%3A%2F%2Fkenn3o3.github.io%2F"><img src="https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fkenn3o3.github.io%2F&labelColor=%23d9e3f0&countColor=%232ccce4" /></a>

## News
- (2026-09-15) Our survey, [**A Survey of Robotic Navigation and Manipulation with Physics Simulators in the Era of Embodied AI**](/projects/eai-survey), has been accepted to **ACM Computing Surveys (CSUR)**
- (2026-09-15) Our paper, [**Equivariant Visual-Tactile Diffusion Policy for Contact-Rich Manipulation**](https://vista-paper.github.io/), has been accepted to **CoRL 2026**
- (2025-06-04) Graduated from **City University of Hong Kong (CityUHK)** with **BSc in Computer Science**
- (2024-11-01) Admitted to **The Chinese University of Hong Kong (CUHK)** MSc in Computer Science (2025 Fall Entry)

## Project Highlights
Below are some of my research projects:

<table class="project-highlights">
  {% assign sorted_projects = site.projects | sort: 'highlight_order' | reverse %}
  {% for project in sorted_projects %}
    <tr class="project-row">
    <td class="project-thumbnail-cell">
      <div class="project-thumbnail">
        <img src="{{ project.thumbnail }}" class="project-thumbnail-image" alt="">
      </div>
    </td>
    <td class="project-details">
      <a href="{{ project.url }}" class="project-title">{{ project.title }}</a>
      <div class="project-authors">{{ project.authors }}</div>
      <div class="project-meta"><em class="project-venue">{{ project.venue }}</em><span class="project-date"> · {{ project.date | date: "%Y-%m-%d" }}</span></div>
      <div class="project-links">
        {% if project.project_url %}<a href="{{ project.project_url }}" target="_blank">Project Page</a>{% endif %}
        {% if project.paper_url %}<a href="{{ project.paper_url }}" target="_blank">Paper</a>{% endif %}
        {% if project.poster_url %}<a href="{{ project.poster_url }}" target="_blank">Poster</a>{% endif %}
        {% if project.code_url %}<a href="{{ project.code_url }}" target="_blank">Code</a>{% endif %}
        {% if project.arxiv_url %}<a href="{{ project.arxiv_url }}" target="_blank">arXiv</a>{% endif %}
      </div>
    </td>
  </tr>
  {% endfor %}
</table>

## Teaching

{% assign teaching_items = site.teaching | sort: 'date' | reverse %}
{% for item in teaching_items %}
- **{{ item.title }}** — {{ item.type }}, {{ item.date | date: "%Y %b" }}
{% endfor %}
