#!/usr/bin/env python3
"""
Version Manager per Crystal Therapy Video Generator
Crea automaticamente tag Git per ogni video generato
"""

import subprocess
import os
import sys
from datetime import datetime
import re

class VersionManager:
    def __init__(self, repo_path="."):
        self.repo_path = repo_path
        self.is_git_repo = self._check_git_repo()
    
    def _check_git_repo(self):
        """Verifica se siamo in un repository Git"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"], 
                cwd=self.repo_path, 
                capture_output=True, 
                text=True, 
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _run_git_command(self, command):
        """Esegue un comando Git e ritorna il risultato"""
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip(), result.stderr.strip()
        except subprocess.CalledProcessError as e:
            return None, e.stderr.strip()
    
    def _sanitize_tag_name(self, video_filename):
        """Sanitizza il nome del video per creare un tag valido"""
        # Rimuove l'estensione .mp4
        tag_name = video_filename.replace('.mp4', '')
        
        # Sostituisce caratteri non validi per i tag Git
        tag_name = re.sub(r'[^a-zA-Z0-9._-]', '_', tag_name)
        
        # Rimuove underscore multipli consecutivi
        tag_name = re.sub(r'_+', '_', tag_name)
        
        # Rimuove underscore all'inizio e alla fine
        tag_name = tag_name.strip('_')
        
        return tag_name
    
    def get_current_commit_hash(self):
        """Ottiene l'hash del commit corrente"""
        if not self.is_git_repo:
            return None
        
        stdout, stderr = self._run_git_command(["git", "rev-parse", "HEAD"])
        return stdout if stdout else None
    
    def get_current_branch(self):
        """Ottiene il branch corrente"""
        if not self.is_git_repo:
            return None
        
        stdout, stderr = self._run_git_command(["git", "branch", "--show-current"])
        return stdout if stdout else None
    
    def check_working_tree_clean(self):
        """Verifica se l'albero di lavoro è pulito (nessun file modificato)"""
        if not self.is_git_repo:
            return False
        
        stdout, stderr = self._run_git_command(["git", "status", "--porcelain"])
        return len(stdout) == 0
    
    def commit_changes(self, message):
        """Fa commit di tutti i cambiamenti"""
        if not self.is_git_repo:
            return False, "Non in un repository Git"
        
        # Aggiungi tutti i file modificati
        stdout, stderr = self._run_git_command(["git", "add", "."])
        if stdout is None:
            return False, f"Errore add: {stderr}"
        
        # Commit
        stdout, stderr = self._run_git_command(["git", "commit", "-m", message])
        if stdout is None:
            return False, f"Errore commit: {stderr}"
        
        return True, stdout
    
    def tag_exists(self, tag_name):
        """Verifica se un tag esiste già"""
        if not self.is_git_repo:
            return False
        
        stdout, stderr = self._run_git_command(["git", "tag", "-l", tag_name])
        return len(stdout) > 0
    
    def create_tag(self, tag_name, message):
        """Crea un tag annotato"""
        if not self.is_git_repo:
            return False, "Non in un repository Git"
        
        if self.tag_exists(tag_name):
            return False, f"Tag {tag_name} esiste già"
        
        stdout, stderr = self._run_git_command(["git", "tag", "-a", tag_name, "-m", message])
        if stdout is None:
            return False, f"Errore creazione tag: {stderr}"
        
        return True, f"Tag {tag_name} creato con successo"
    
    def push_tag(self, tag_name):
        """Push del tag su origin"""
        if not self.is_git_repo:
            return False, "Non in un repository Git"
        
        # Prima controlla se siamo in detached HEAD
        current_branch = self.get_current_branch()
        if not current_branch:
            # In detached HEAD, prima spostiamoci sul branch principale
            stdout, stderr = self._run_git_command(["git", "switch", "-c", "main", "HEAD"])
            if stdout is None:
                # Se main esiste già, prova a switchare
                stdout, stderr = self._run_git_command(["git", "switch", "main"])
                if stdout is None:
                    return False, f"Impossibile uscire da detached HEAD: {stderr}"
        
        stdout, stderr = self._run_git_command(["git", "push", "origin", tag_name])
        if stdout is None:
            return False, f"Errore push tag: {stderr}"
        
        return True, f"Tag {tag_name} pushato su origin"
    
    def create_version_for_video(self, video_filename, config_summary=None):
        """
        Crea una versione completa per un video generato:
        1. Commit delle modifiche (se necessario)
        2. Creazione del tag
        3. Push del tag (opzionale)
        """
        if not self.is_git_repo:
            print("⚠️  Non siamo in un repository Git. Saltando il versionamento.")
            return False
        
        # Sanitizza il nome del tag
        tag_name = self._sanitize_tag_name(video_filename)
        
        print(f"🏷️  Creando versione per video: {video_filename}")
        print(f"📋 Tag Git: {tag_name}")
        
        # Controlla se ci sono modifiche da committare
        if not self.check_working_tree_clean():
            print("📝 Rilevate modifiche non committate. Facendo commit...")
            commit_message = f"Configurazione per video {video_filename}"
            if config_summary:
                commit_message += f"\n\n{config_summary}"
            
            success, message = self.commit_changes(commit_message)
            if not success:
                print(f"❌ Errore nel commit: {message}")
                return False
            print(f"✅ Commit completato: {message}")
        
        # Crea il tag
        tag_message = f"Video generato: {video_filename}"
        if config_summary:
            tag_message += f"\n\nConfigurazione:\n{config_summary}"
        
        # Aggiungi informazioni tecniche al tag
        current_commit = self.get_current_commit_hash()
        current_branch = self.get_current_branch()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        tag_message += f"\n\nDettagli tecnici:"
        tag_message += f"\nCommit: {current_commit[:8] if current_commit else 'N/A'}"
        tag_message += f"\nBranch: {current_branch or 'N/A'}"
        tag_message += f"\nData: {timestamp}"
        
        success, message = self.create_tag(tag_name, tag_message)
        if not success:
            print(f"❌ Errore nella creazione del tag: {message}")
            return False
        
        print(f"✅ {message}")
        
        # Prova il push del tag (opzionale, non fallire se non riesce)
        print("📤 Tentativo push del tag su origin...")
        success, message = self.push_tag(tag_name)
        if success:
            print(f"✅ {message}")
            print(f"🌐 Tag disponibile su GitHub: {tag_name}")
        else:
            print(f"⚠️  Push non riuscito (normale se non hai configurato origin): {message}")
        
        print(f"🔄 Per rigenerare questo video: git checkout {tag_name}")
        return True

def main():
    """Funzione principale per test"""
    vm = VersionManager()
    
    if len(sys.argv) < 2:
        print("Uso: python version_manager.py <nome_video.mp4> [descrizione_config]")
        sys.exit(1)
    
    video_filename = sys.argv[1]
    config_summary = sys.argv[2] if len(sys.argv) > 2 else None
    
    vm.create_version_for_video(video_filename, config_summary)

if __name__ == "__main__":
    main()
