---
permalink: /
title: "Kenny Wong — Embodied AI and Robotics"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

<section class="home-hero" aria-labelledby="home-title">
  <p class="home-kicker"><span aria-hidden="true"></span>MPhil at CUHK · Embodied AI &amp; Robotics</p>
  <h1 id="home-title">I build learning systems for robots that <em>see, feel, and act.</em></h1>
  <p class="home-lead">I am <strong>Kenny Wong Lik Hang</strong> (王力恒), an MPhil student in Computer Science and Engineering at The Chinese University of Hong Kong, advised by <a href="https://www.cse.cuhk.edu.hk/~qdou/">Prof. Dou Qi</a>. My research focuses on visuotactile perception and policy learning for contact-rich manipulation.</p>
  <p class="home-background">I received my MSc in Computer Science and Engineering from CUHK in 2026 and my BSc in Computer Science from City University of Hong Kong in 2025.</p>
  <div class="home-actions">
    <a class="home-button home-button--primary" href="#selected-work">Selected work <span aria-hidden="true">↓</span></a>
    <a class="home-button" href="/files/resume.pdf">Resume <span aria-hidden="true">↗</span></a>
    <a class="home-button" href="mailto:klhwong3@outlook.com">Email</a>
  </div>
</section>

<div class="home-section-heading">
  <div><p>Recent</p><h2>News</h2></div>
  <span>Research and academic updates</span>
</div>

<ul class="news-feed">
  <li class="news-item--paper">
    <time datetime="2026-09-15">Sep 2026</time>
    <div>Our survey, <a href="/projects/eai-survey">A Survey of Robotic Navigation and Manipulation with Physics Simulators in the Era of Embodied AI</a>, has been accepted to <em>ACM Computing Surveys (CSUR)</em>.</div>
  </li>
  <li class="news-item--paper">
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

<div class="home-section-heading" id="selected-work">
  <div><p>Research</p><h2>Selected work</h2></div>
  <a href="/projects/">View all projects <span aria-hidden="true">↗</span></a>
</div>

{% include project-list.html %}

<div class="home-section-heading">
  <div><p>Background</p><h2>Experience &amp; highlights</h2></div>
</div>

<div class="experience-grid">
  <article class="experience-card">
    <div class="experience-card__top"><span>Research</span><time>May - Aug 2024</time></div>
    <h3>Oak Ridge National Laboratory · UTK</h3>
    <p class="experience-role">Undergraduate Researcher, NICS</p>
    <p>Developed topological-map navigation methods for legged robot image-goal navigation using Habitat-Sim and Isaac Sim. The research was supported by the National Science Foundation and supervised by Dr. Kwai Wong.</p>
  </article>
  <article class="experience-card">
    <div class="experience-card__top"><span>Industry</span><time>Jul 2023 - May 2024</time></div>
    <h3>The Bank of East Asia</h3>
    <p class="experience-role">Student Programmer, IT Department</p>
    <p>Worked on backend automation and internal software systems using Java, SQL, Linux, sockets, Selenium, and Git within the Technology Innovation and IT Project Management section.</p>
  </article>
  <article class="experience-card experience-card--accent">
    <div class="experience-card__top"><span>Recognition</span><time>2023 - 2025</time></div>
    <h3>Selected achievements</h3>
    <ul>
      <li><strong>2nd of 23</strong> in the WACV 2025 Elderly Action Recognition Challenge.</li>
      <li><strong>Top 5.4%</strong> worldwide in the IEEEXtreme 18.0 programming contest.</li>
      <li><strong>1st</strong> in AUC and accuracy for a DeepFake detection course project.</li>
    </ul>
  </article>
</div>

<div class="home-section-heading">
  <div><p>Academic service</p><h2>Teaching</h2></div>
</div>

<ul class="teaching-list">
{% assign teaching_items = site.teaching | sort: 'date' | reverse %}
{% for item in teaching_items %}
  <li>
    <a class="teaching-title" href="{{ item.url | relative_url }}">{{ item.title }}</a>
    <div class="teaching-meta">{{ item.type }} · {{ item.venue }} · {{ item.date | date: "%b %Y" }}</div>
    {% if item.excerpt %}<p class="teaching-excerpt">{{ item.excerpt }}</p>{% endif %}
  </li>
{% endfor %}
</ul>
