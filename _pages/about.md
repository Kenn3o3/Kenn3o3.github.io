---
permalink: /
title: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I am **Wong Lik Hang Kenny** (Chinese name: **王力恒**), a recent graduate with a **BSc in Computer Science** from **City University of Hong Kong (CityUHK)** in 2025. I am excited to continue my academic journey in the **MSc in Computer Science** program at **The Chinese University of Hong Kong (CUHK)** starting in Fall 2025.

My research interests lie at the intersection of Deep Learning, 3D Computer Vision, Embodied AI, Robotics, and Continual Learning. I am currently working on continual learning (lifelong learning), which I believe is essential for developing generalist Embodied AI Agents capable of functioning in real-world environments.

***"Life is about adapting to challenges and remembering the lessons they teach."***

I started writing blogs recently.

You can view my CV ([here](/files/resume.pdf)). Feel free to reach out to me at `klhwong3 [at] outlook [dot] com`—I’d love to collaborate on something exciting in the future! :>

## News
- (2025-06-04) Graduated from **City University of Hong Kong (CityUHK)** with **BSc in Computer Science**
- (2024-11-01) Admitted to **The Chinese University of Hong Kong (CUHK)** MSc in Computer Science (2025 Fall Entry)

## Project Highlights
Below are some of my research projects:

<table style="width: 100%; border: none;">
  {% assign sorted_projects = site.projects | sort: 'date' | reverse %}
  {% for project in sorted_projects %}
    <tr style="width: 100%; border: none;">
    <td width="20%" style="padding: 10px 30px 10px 10px; border: none;">
      <div class="container">
        <img src="{{ project.thumbnail }}" width="180px" style="box-shadow: 4px 4px 4px #888888; margin-left: 10px;">
      </div>
    </td>
    <td style="padding: 10px 30px 10px 10px; border: none;">
      <a href="{{ project.url }}" style="font-size: 15px; font-weight: bold;">{{ project.title }}</a>
      <div style="height: 5px;"></div>
      <div style="font-size: 12px">{{ project.authors }}</div>
      <div style="height: 5px;"></div>
      <div style="font-size: 12px">{{ project.venue }}, {{ project.date | date: "%Y-%m-%d" }}</div>
      <div style="height: 5px;"></div>
      <div style="font-size: 12px">
        {% if project.paper_url %}<a href="{{ project.paper_url }}" target="_blank">Paper</a> / {% endif %}
        {% if project.poster_url %}<a href="{{ project.poster_url }}" target="_blank">Poster</a> / {% endif %}
        {% if project.code_url %}<a href="{{ project.code_url }}" target="_blank">Code</a> / {% endif %}
        {% if project.arxiv_url %}<a href="{{ project.arxiv_url }}" target="_blank">arXiv</a>{% endif %}
      </div>
    </td>
  </tr>
  {% endfor %}
</table>

## My Corner

### Blogs
<table style="width: 100%; border: none;">
  {% assign sorted_blogs = site.blogs | sort: 'date' | reverse %}
  {% for blog in sorted_blogs %}
    <tr>
      <td style="border: none;">
        <a href="{{ blog.url }}" style="font-size: 15px; font-weight: bold;">{{ blog.title }} ({{ blog.date | date: "%Y-%m-%d" }})</a>
        <div style="height: 3px;"></div>
      </td>
    </tr>
  {% endfor %}
</table>

### Learning
<table style="width: 100%; border: none;">
  {% assign sorted_learning = site.learning | sort: 'date' | reverse %}
  {% for learning in sorted_learning %}
    <tr>
      <td style="border: none;">
        <a href="{{ learning.url }}" style="font-size: 15px; font-weight: bold;">{{ learning.title }} ({{ learning.date | date: "%Y-%m-%d" }})</a>
        <div style="height: 3px;"></div>
      </td>
    </tr>
  {% endfor %}
</table>

### Courses

(In Construction)