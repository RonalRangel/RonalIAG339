function downloadPDF() {
  const element = document.querySelector('#pdf-content');

  const hv = {
    margin: [0, 0, 0, 0],  // [arriba, izquierda, abajo, derecha] en mm
    filename: 'Hoja_de_Vida_Julian_Buitrago.pdf',
    image: { type: 'jpeg', quality: 0.98 },
    html2canvas: {
      scale: 2,
      useCORS: true,
      allowTaint: false,
      scrollY: 0
    },
    jsPDF: {
      unit: 'mm',
      format: 'a4',
      orientation: 'portrait'
    }
  };

  html2pdf().set(hv).from(element).save();
}