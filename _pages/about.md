---
permalink: /
title: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I am **Kenny Wong Lik Hang** (王力恒), an MPhil student in Computer Science and Engineering at The Chinese University of Hong Kong (CUHK), advised by [Prof. Dou Qi](https://www.cse.cuhk.edu.hk/~qdou/). I received my MSc in Computer Science and Engineering from CUHK in 2026, and my BSc in Computer Science from City University of Hong Kong (CityUHK) in 2025.

My research focuses on **Embodied AI and Robotics**, especially visuotactile perception and policy learning for contact-rich manipulation.

You can find my [CV](/files/resume.pdf). Feel free to reach me at klhwong3 [at] outlook [dot] com — I would love to chat about research or collaboration.

## News

<ul class="news-feed">
  <li>
    <time datetime="2026-09-15">Sep 2026</time>
    <div>Our survey, <a href="/projects/eai-survey">A Survey of Robotic Navigation and Manipulation with Physics Simulators in the Era of Embodied AI</a>, has been accepted to <em>ACM Computing Surveys (CSUR)</em>.</div>
  </li>
  <li>
    <time datetime="2026-09-15">Sep 2026</time>
    <div>Our paper, <a href="https://vista-paper.github.io/">Equivariant Visual-Tactile Diffusion Policy for Contact-Rich Manipulation</a>, has been accepted to <em>Conference on Robot Learning (CoRL 2026)</em>.</div>
  </li>
  <li>
    <time datetime="2025-06-04">Jun 2025</time>
    <div>Graduated from City University of Hong Kong (CityUHK) with a BSc in Computer Science.</div>
  </li>
  <li>
    <time datetime="2024-11-01">Nov 2024</time>
    <div>Admitted to the MSc in Computer Science programme at The Chinese University of Hong Kong (CUHK), 2025 Fall Entry.</div>
  </li>
</ul>

## Selected Projects

{% include project-list.html %}

## Teaching

<ul class="teaching-list">
{% assign teaching_items = site.teaching | sort: 'date' | reverse %}
{% for item in teaching_items %}
  <li>
    <a class="teaching-title" href="{{ item.url | relative_url }}">{{ item.title }}</a>
    <div class="teaching-meta">{{ item.type }} · {{ item.venue }} · {{ item.date | date: "%b %Y" }}</div>
  </li>
{% endfor %}
</ul>
