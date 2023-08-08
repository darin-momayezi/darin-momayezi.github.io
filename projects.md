---
layout: "page"
title: "Projects"
---

{% if site.show_excerpts %}
  {% include projects.html %}
{% else %}
  {% include archive.html title="Posts" %}
{% endif %}
