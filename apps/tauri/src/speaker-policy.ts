/** Speaker IDs share RuntimeConfig's signed INT32 representation across backends. */
export function speakerIdMaximum(speakerCount?: number | null): number {
  return typeof speakerCount === 'number' && Number.isSafeInteger(speakerCount) && speakerCount > 0
    ? Math.min(speakerCount - 1, 2_147_483_647)
    : 2_147_483_647;
}

export function validateSpeakerId(speakerId: number, speakerCount?: number | null): string | null {
  if (!Number.isInteger(speakerId) || speakerId < 0 || speakerId > 2_147_483_647) {
    return 'Speaker ID must be a whole number between 0 and 2147483647.';
  }
  const maximum = speakerIdMaximum(speakerCount);
  if (speakerId > maximum) {
    return `This model supports speaker IDs 0–${maximum}. Choose a valid speaker ID.`;
  }
  return null;
}
