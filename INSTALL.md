# تثبيت Fodci وتشغيله من أي Terminal

> **الحالة الحالية:** أمر `fodci` يعمل كواجهة محلية للنموذج المستقر `fodci-testing-qa-v1.pt`. النموذج صغير ومحلي، لذلك لا ينبغي اعتبار جودة الردود الحالية مساوية لـ ChatGPT أو Manus. لا توجد خدمة سحابية أو تدريب Online ضمن هذا التثبيت.

## Windows — الطريقة الموصى بها

تحتاج إلى **Python 3.12**. وجود Python 3.15 وحده غير كافٍ لأن بيئة المشروع المستقرة تستهدف Python 3.11 و3.12.

افتح PowerShell ونفّذ الأوامر التالية:

```powershell
git clone https://github.com/peterisaac-00/Fodci_Ai.git
cd Fodci_Ai
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\install_fodci_global.ps1
```

يقوم السكربت بإنشاء بيئة مستقلة في `$env:USERPROFILE\.fodci\venv`، يثبت المشروع بصيغة editable مع extra النموذج CPU، ينزّل checkpoint المستقر من GitHub Release ويتحقق من SHA-256، ثم يضيف مجلد `Scripts` إلى **User PATH**. بعد انتهاء السكربت أغلق PowerShell وافتح نافذة جديدة، ثم شغّل:

```powershell
fodci
```

بعد ذلك لا تحتاج إلى تفعيل `.venv` داخل كل مشروع؛ الأمر `fodci` متاح من أي مجلد. لتأكيد مكان الأمر:

```powershell
Get-Command fodci
```

ولإيقاف الجلسة استخدم `/exit`.

### التثبيت اليدوي على Windows

إذا أردت تنفيذ الخطوات يدويًا بدل السكربت:

```powershell
py -3.12 -m venv "$env:USERPROFILE\.fodci\venv"
& "$env:USERPROFILE\.fodci\venv\Scripts\python.exe" -m pip install --upgrade pip
& "$env:USERPROFILE\.fodci\venv\Scripts\python.exe" -m pip install -e ".[model]"
.\scripts\download_phase1312_checkpoint.ps1
```

أضف هذا المسار إلى User PATH مرة واحدة من إعدادات Windows:

```text
%USERPROFILE%\.fodci\venv\Scripts
```

ثم افتح Terminal جديدة وشغّل `fodci`.

## Linux وmacOS

المشروع الأساسي لا يفرض runtime dependencies. لتشغيل النموذج المحلي، ثبّت Python 3.11 أو 3.12 وPyTorch عبر extra `model`:

```bash
git clone https://github.com/peterisaac-00/Fodci_Ai.git
cd Fodci_Ai
python3.12 -m venv "$HOME/.fodci/venv"
source "$HOME/.fodci/venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e '.[model]'
```

نزّل checkpoint المستقر إلى المسار الذي يتوقعه التطبيق:

```bash
mkdir -p artifacts/checkpoints
curl -fL "https://github.com/peterisaac-00/Fodci_Ai/releases/download/v13.12-stable/fodci-testing-qa-v1.pt" \
  -o artifacts/checkpoints/fodci-testing-qa-v1.pt
```

تحقق من SHA-256 قبل التشغيل. القيمة الصحيحة هي:

```text
3af5d2b6009f5a0fd0ff98644d9666bd0c30f0dfe8994f91524ae6df11433bfa
```

على Linux:

```bash
sha256sum artifacts/checkpoints/fodci-testing-qa-v1.pt
```

وعلى macOS:

```bash
shasum -a 256 artifacts/checkpoints/fodci-testing-qa-v1.pt
```

لجعل الأمر متاحًا دائمًا من أي Terminal، أضف البيئة إلى shell profile:

```bash
echo 'export PATH="$HOME/.fodci/venv/bin:$PATH"' >> "$HOME/.bashrc"
# macOS أو zsh:
echo 'export PATH="$HOME/.fodci/venv/bin:$PATH"' >> "$HOME/.zshrc"
```

افتح Terminal جديدة ثم نفّذ:

```bash
fodci
```

## التحديث بعد `git pull`

لأن التثبيت editable يشير مباشرة إلى نسخة المستودع، حدّث الكود ثم حدّث package metadata عند الحاجة:

```powershell
# Windows
cd C:\path\to\Fodci_Ai
git pull origin main
& "$env:USERPROFILE\.fodci\venv\Scripts\python.exe" -m pip install -e ".[model]"
```

```bash
# Linux/macOS
cd /path/to/Fodci_Ai
git pull origin main
python -m pip install -e '.[model]'
```

إذا ظهر إصدار جديد من checkpoint، أعد تشغيل سكربت التنزيل في Windows أو نزّل ملف الإصدار الجديد يدويًا بعد التحقق من قيمة SHA-256 المنشورة في توثيق الإصدار. لا تستبدل `fodci-testing-qa-v1.pt` بcheckpoint تجريبي من Phase 15؛ candidate distillation لا تُرقّى تلقائيًا.

## تثبيت أدوات التطوير فقط

إذا كنت تريد تشغيل الاختبارات دون تثبيت PyTorch، استخدم البيئة الأساسية مع extra التطوير:

```bash
python -m pip install -e '.[dev]'
PYTHONPATH=src:. pytest -q
```

هذا الوضع لا يكفي لتشغيل `fodci` مع checkpoint المحلي؛ لتشغيل التطبيق استخدم `.[model]` وتأكد من وجود checkpoint المستقر.
