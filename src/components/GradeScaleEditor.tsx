import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card } from '@/components/ui/card';
import { Plus, Trash2 } from 'lucide-react';

interface GradeScaleEditorProps {
  value: Record<string, number>;
  onChange: (value: Record<string, number>) => void;
}

export function GradeScaleEditor({ value, onChange }: GradeScaleEditorProps) {
  const [entries, setEntries] = useState(
    Object.entries(value).map(([grade, points]) => ({ grade, points }))
  );

  useEffect(() => {
    setEntries(Object.entries(value).map(([grade, points]) => ({ grade, points })));
  }, [value]);

  const handleUpdate = (index: number, field: 'grade' | 'points', newValue: string) => {
    const updated = [...entries];
    if (field === 'grade') {
      updated[index].grade = newValue;
    } else {
      updated[index].points = parseFloat(newValue) || 0;
    }
    setEntries(updated);
    
    const obj: Record<string, number> = {};
    updated.forEach(({ grade, points }) => {
      if (grade) obj[grade] = points;
    });
    onChange(obj);
  };

  const handleAdd = () => {
    const updated = [...entries, { grade: '', points: 0 }];
    setEntries(updated);
  };

  const handleRemove = (index: number) => {
    const updated = entries.filter((_, i) => i !== index);
    setEntries(updated);
    
    const obj: Record<string, number> = {};
    updated.forEach(({ grade, points }) => {
      if (grade) obj[grade] = points;
    });
    onChange(obj);
  };

  return (
    <Card className="p-4 space-y-4 bg-card/50 backdrop-blur-sm border-border/50">
      <div className="flex items-center justify-between">
        <Label className="text-base font-semibold text-foreground">Grade Scale Configuration</Label>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={handleAdd}
          className="gap-2"
        >
          <Plus className="h-4 w-4" />
          Add Grade
        </Button>
      </div>

      <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
        {entries.map((entry, index) => (
          <div key={index} className="flex gap-2 items-center group">
            <Input
              placeholder="Grade (e.g., A+)"
              value={entry.grade}
              onChange={(e) => handleUpdate(index, 'grade', e.target.value)}
              className="flex-1 bg-background/50"
            />
            <Input
              type="number"
              placeholder="Points"
              value={entry.points}
              onChange={(e) => handleUpdate(index, 'points', e.target.value)}
              step="0.01"
              min="0"
              max="10"
              className="w-24 bg-background/50"
            />
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => handleRemove(index)}
              className="opacity-0 group-hover:opacity-100 transition-opacity text-destructive hover:text-destructive"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        ))}
      </div>
    </Card>
  );
}
