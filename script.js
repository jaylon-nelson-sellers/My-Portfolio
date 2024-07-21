document.addEventListener('DOMContentLoaded', (event) => {
    fetch('data.json')
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById('data-container');
            container.innerHTML = `<h2>Python-generated Data:</h2>
                                   <p>Random number: ${data.random_number}</p>
                                   <p>Current date: ${data.current_date}</p>`;
        })
        .catch(error => console.error('Error:', error));
});