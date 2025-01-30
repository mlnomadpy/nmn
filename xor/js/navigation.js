// Navigation handling
document.querySelectorAll('.nav-item[data-view]').forEach(item => {
  item.addEventListener('click', () => {
    // Update navigation active state
    document.querySelectorAll('.nav-item[data-view]').forEach(nav => nav.classList.remove('active'));
    item.classList.add('active');
    
    // Show selected view
    document.querySelectorAll('.view').forEach(view => view.classList.remove('active'));
    document.getElementById(`${item.dataset.view}-view`).classList.add('active');
  });
});

// Neuron type selection handling
document.querySelectorAll('.nav-item[data-neuron]').forEach(item => {
  item.addEventListener('click', () => {
    // Update neuron selection active state
    document.querySelectorAll('.nav-item[data-neuron]').forEach(nav => nav.classList.remove('active'));
    item.classList.add('active');
    
    // Show selected neuron
    document.querySelectorAll('.neuron-card').forEach(card => {
      card.style.display = card.id.startsWith(item.dataset.neuron) ? 'block' : 'none';
    });
  });
});
