// Initialize GSAP ScrollTrigger
gsap.registerPlugin(ScrollTrigger);

// Animate sections on scroll
gsap.utils.toArray('.section').forEach(section => {
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

// Initialize Swiper for project slider
const projectSlider = new Swiper('.project-slider', {
    slidesPerView: 1,
    spaceBetween: 30,
    loop: true,
    pagination: {
        el: '.swiper-pagination',
        clickable: true,
    },
    navigation: {
        nextEl: '.swiper-button-next',
        prevEl: '.swiper-button-prev',
    },
    breakpoints: {
        640: {
            slidesPerView: 2,
        },
        1024: {
            slidesPerView: 3,
        },
    },
});

// Project data
const projects = [
    {
        title: "Global Country Analysis",
        description: "Applying unsupervised learning techniques to analyze countries worldwide.",
        image: "media/global.jpg",
        tags: ["Clustering", "Data Visualization", "Geo-politics"],
    },
    {
        title: "Text Analysis: Predicting online reactions",
        description: "Analyzing online posts using unsupervised and supervised learning techniques.",
        image: "media/AITA.jpg",
        tags: ["Natural Language Processing", "Clustering", "Classification"],
    },
    {
        title: "Video Game Analysis: Pokemon Unite",
        description: "Discovering trends in powerful characters using unsupervised learning.",
        image: "media/unite.jpg",
        tags: ["Video Games/Esports", "Clustering", "Data Visualization"],
    },
];

// Populate project slider
function populateProjectSlider() {
    const swiperWrapper = document.querySelector('.swiper-wrapper');
    projects.forEach(project => {
        const slide = document.createElement('div');
        slide.className = 'swiper-slide';
        slide.innerHTML = `
            <div class="project-card">
                <div class="project-image" style="background-image: url('${project.image}');"></div>
                <div class="project-content">
                    <h3>${project.title}</h3>
                    <p>${project.description}</p>
                    <div class="tags">
                        ${project.tags.map(tag => `<span>${tag}</span>`).join('')}
                    </div>
                    <a href="#" class="btn">View Project (Coming Soon)</a>
                </div>
            </div>
        `;
        swiperWrapper.appendChild(slide);
    });
    projectSlider.update();
}

// Call the function to populate the slider
populateProjectSlider();

// Smooth scroll for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});