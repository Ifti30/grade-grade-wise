
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import React from 'react';

interface CustomFileInputProps {
  id: string;
  file: File | null;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  accept?: string;
  className?: string;
}

export function CustomFileInput({ id, file, onFileChange, accept, className }: CustomFileInputProps) {
  return (
    <div className={className}>
      <Input
        id={id}
        type="file"
        accept={accept}
        onChange={onFileChange}
        className="hidden"
      />
      <div className="flex items-center w-full h-11 rounded-md border border-input bg-background/50 text-sm overflow-hidden">
        <Label
          htmlFor={id}
          className="flex items-center h-full px-4 bg-primary text-primary-foreground cursor-pointer font-medium whitespace-nowrap hover:bg-primary/90"
        >
          Choose File
        </Label>
        <span className={`px-4 overflow-hidden text-ellipsis whitespace-nowrap ${file ? 'text-foreground' : 'text-muted-foreground'}`}>
          {file ? file.name : 'No file chosen'}
        </span>
      </div>
    </div>
  );
}
