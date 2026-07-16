export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.download = filename;
  link.href = url;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadBytes(bytes: Uint8Array, filename: string, mimeType = 'application/octet-stream'): void {
  const blob = new Blob([bytes], { type: mimeType });
  downloadBlob(blob, filename);
}
