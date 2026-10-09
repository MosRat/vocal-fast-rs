export const SESSION_PROFILES = [
  { id: 'reference', name: 'Reference', description: 'Flexible inference window', detail: '11,200 – 44,960 samples' },
  { id: 'voice_changer', name: 'Voice changer', description: 'Compact inference window', detail: '11,200 samples' },
  { id: 'realtime_44960', name: 'Applio', description: 'Full realtime context', detail: '44,960 samples' },
] as const;

export type SessionProfile = typeof SESSION_PROFILES[number]['id'];

export function renderSessionProfiles(selected: string, disabled: boolean, prefix = ''): string {
  return `<fieldset class="session-profiles" ${disabled ? 'disabled' : ''}>
    <legend class="field-label">Session profile</legend>
    <div class="profile-options">${SESSION_PROFILES.map((profile) => `<label class="profile-option ${selected === profile.id ? 'selected' : ''}">
      <input id="${prefix}profile-${profile.id}" type="radio" name="${prefix}sessionProfile" value="${profile.id}" data-session-profile ${selected === profile.id ? 'checked' : ''}/>
      <span class="profile-option-copy"><strong>${profile.name}</strong><span>${profile.description}</span><small>${profile.detail}</small></span>
    </label>`).join('')}</div>
  </fieldset>`;
}
