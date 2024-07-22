// Initialize GSAP ScrollTrigger
gsap.registerPlugin(ScrollTrigger);

// Animate sections on scroll
gsap.utils.toArray('section').forEach(section => {
    gsap.from(section, {
        opacity: 0,
        y: 50,
        duration: 1,
        scrollTrigger: {
            trigger: section,
            start: "top 80%",
            end: "top 20%",
            scrub: 1
        }
    });
});

// Animate skill items
gsap.from('.skill-category', {
    scale: 0.9,
    opacity: 0,
    duration: 0.5,
    stagger: 0.1,
    scrollTrigger: {
        trigger: '.skills-container',
        start: "top 80%"
    }
});

// Smooth scroll for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Interactive skill list
document.querySelectorAll('.skill-category li').forEach(skill => {
    skill.addEventListener('click', function() {
        this.style.color = this.style.color === 'var(--accent-color)' ? '' : 'var(--accent-color)';
    });
});